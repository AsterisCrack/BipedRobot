"""BipedEnvV2 — MuJoCo drop-in inference target for Isaac Lab BipedRobotV2 policies.

Observation (50 dims) matches Isaac Lab BipedRobotV2EnvCfg exactly:
  lin_acc_b(3) | ang_vel_b(3) | projected_gravity(3) | commands(3)
  | joint_pos_rel(12) | joint_vel(12) | prev_actions(12) | gait_phase(2)

Privileged / critic obs (60 dims):
  obs_50 + base_lin_vel_b(3) + height(1) + feet_forces(6)

Action: 12 joints in [-1, 1], mapped to joint limits via the same formula as Isaac Lab.
Control loop: 50 Hz (decimation=4 at 200 Hz physics), matching Isaac Lab.
"""
from __future__ import annotations

import math
import os
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium.spaces import Box, Dict

from envs.mujoco.base_env import BaseEnv
from envs.utils.randomizer import Randomizer
from envs.isaaclab.rewards import rewards as R
from utils import NoConfig

# ---------------------------------------------------------------------------
# Joint layout — MUST match Isaac Lab BipedRobotV2EnvCfg (L,R interleaved)
#   idx:  0         1         2                3                4                5
#         l_hip_yaw r_hip_yaw l_hip_roll_joint r_hip_roll_joint l_hip_pitch_joint r_hip_pitch_joint
#   idx:  6           7         8                9                10               11
#         l_knee_joint r_knee_joint l_ankle_roll_joint r_ankle_roll_joint l_ankle_pitch_joint r_ankle_pitch_joint
# ---------------------------------------------------------------------------
JOINT_NAMES: list[str] = [
    "l_hip_yaw",           "r_hip_yaw",
    "l_hip_roll_joint",    "r_hip_roll_joint",
    "l_hip_pitch_joint",   "r_hip_pitch_joint",
    "l_knee_joint",        "r_knee_joint",
    "l_ankle_roll_joint",  "r_ankle_roll_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint",
]

# Joint soft limits from Isaac Lab V2 (radians)
JLIM_MIN = np.array([
    -0.785, -0.785,   # hip yaw  ±45°
    -0.436, -0.436,   # hip roll  -25°…+45°
    -1.571, -1.571,   # hip pitch -90°…+30°
    -2.094, -2.094,   # knee      -120°…+5°
    -1.047, -1.047,   # ankle roll ±60°
    -0.611, -0.611,   # ankle pitch -35°…+50°
], dtype=np.float32)

JLIM_MAX = np.array([
    0.785,  0.785,
    0.785,  0.785,
    0.524,  0.524,
    0.087,  0.087,
    1.047,  1.047,
    0.873,  0.873,
], dtype=np.float32)

JRANGE = JLIM_MAX - JLIM_MIN

# Sub-index groups (into the 12-joint vector) for reward functions
HIP_IDX         = [0, 1, 2, 3]  # yaw + roll (Isaac Lab V2 hip_joint_names, pitch excluded)
KNEE_IDX        = [6, 7]
ANKLE_ROLL_IDX  = [8, 9]
ANKLE_PITCH_IDX = [10, 11]

GAIT_CLOCK_FREQ: float = 1.5      # Hz — matches biped_env_cfg.py
ACTION_FILTER_ALPHA: float = 0.4  # EMA — matches biped_env_cfg.py
PHYSICS_DT: float = 0.005         # 200 Hz — matches Isaac Lab sim dt
DECIMATION: int   = 4             # 50 Hz control — matches Isaac Lab decimation
STEP_DT: float    = PHYSICS_DT * DECIMATION

_XML_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../../envs/assets/robotV2/robot_v2.xml",
)

_DEFAULT_CMD_RANGES = {
    "lin_vel_x": (-0.3,  0.5),
    "lin_vel_y": (-0.3,  0.3),
    "ang_vel_z": (-0.3,  0.3),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t1(arr: np.ndarray) -> torch.Tensor:
    """Wrap 1-D numpy array in a [1, N] float32 torch tensor (batch dim=1)."""
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0)


def _t0(val: float) -> torch.Tensor:
    """Wrap a scalar in a [1] float32 torch tensor."""
    return torch.tensor([float(val)], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BipedEnvV2(BaseEnv):
    """MuJoCo V2 biped — drop-in inference target for Isaac Lab-trained policies."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Public attributes expected by model wrappers
    num_envs: int = 1

    def __init__(
        self,
        xml_path: str = _XML_DEFAULT,
        render_mode: str | None = None,
        history_size: int = 0,
        use_history: bool = False,
        critic_has_privileged_info: bool = True,
        env_config=None,
        seed: int | None = None,
    ):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"V2 robot XML not found at {xml_path}.\n"
                "Run  python scripts/generate_v2_xml.py  once to generate it."
            )
        super().__init__(xml_path, render_mode)

        self.config = env_config or NoConfig()
        self.history_size = history_size
        self.use_history  = use_history and history_size > 1
        self.critic_has_privileged_info = critic_has_privileged_info
        self.device = "cpu"

        # Physics timing
        self.model.opt.timestep = PHYSICS_DT

        # ---- Resolve joint / actuator / body IDs -----------------------
        # int() cast: qposadr/dofadr may return a 1-element array in newer MuJoCo
        self._qpos_ids = np.array([int(self.model.joint(n).qposadr) for n in JOINT_NAMES], dtype=int)
        self._qvel_ids = np.array([int(self.model.joint(n).dofadr)  for n in JOINT_NAMES], dtype=int)
        self._act_ids  = np.array([int(self.model.actuator(n).id)   for n in JOINT_NAMES], dtype=int)

        # Foot bodies and their geoms
        self._r_foot_body = self.model.body("r_foot_link_1").id
        self._l_foot_body = self.model.body("l_foot_link_1").id
        self._r_foot_geoms = self._geoms_of_body(self._r_foot_body)
        self._l_foot_geoms = self._geoms_of_body(self._l_foot_body)

        # Floor geom (first plane geom)
        self._floor_geom = next(
            (g for g in range(self.model.ngeom)
             if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE),
            -1
        )

        # Default joint positions (all zeros for V2)
        self._default_joint_pos = np.zeros(12, dtype=np.float32)

        # ---- State ---------------------------------------------------------
        self.commands           = np.zeros(3,  dtype=np.float32)
        self.prev_actions       = np.zeros(12, dtype=np.float32)
        self.filtered_targets   = np.zeros(12, dtype=np.float32)
        self._prev_qvel         = np.zeros(self.model.nv, dtype=np.float32)
        self._last_physics_acc  = np.zeros(3, dtype=np.float64)
        self.gait_phase         = 0.0

        # Gait tracking buffers
        self._in_contact        = np.zeros(2, dtype=bool)    # [right, left]
        self._in_contact_prev   = np.zeros(2, dtype=bool)
        self._air_time          = np.zeros(2, dtype=np.float32)
        self._contact_time      = np.zeros(2, dtype=np.float32)
        self._knee_bend_max     = np.zeros(2, dtype=np.float32)
        self._feet_pos_liftoff  = np.zeros((2, 3), dtype=np.float32)
        self._touchdown         = np.zeros(2, dtype=bool)
        self._both_air_steps    = 0

        # History buffer (policy obs only)
        base_actor_dim  = 50
        base_critic_dim = 60 if critic_has_privileged_info else 50
        self._base_actor_dim  = base_actor_dim
        self._base_critic_dim = base_critic_dim
        if self.use_history:
            self._history = deque(
                [np.zeros(base_actor_dim, dtype=np.float32)] * history_size,
                maxlen=history_size,
            )
        else:
            self._history = None

        # Randomizer
        self._randomizer = Randomizer(self.model, None)

        # Reward weights from config
        self._reward_w: dict[str, float] = {}
        if hasattr(self.config, "reward_weights"):
            self._reward_w = dict(self.config.reward_weights)
        self._reward_scale: float = getattr(self.config, "reward_scale", 1.0)

        # Gymnasium spaces
        actor_dim  = base_actor_dim * (history_size if self.use_history else 1)
        critic_dim = base_critic_dim
        self.observation_space = Dict({
            "actor":  Box(-np.inf, np.inf, shape=(actor_dim,),  dtype=np.float32),
            "critic": Box(-np.inf, np.inf, shape=(critic_dim,), dtype=np.float32),
        })
        self.action_space = Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        if seed is not None:
            np.random.seed(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _geoms_of_body(self, body_id: int) -> list[int]:
        return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body_id]

    def _foot_contact_force(self, foot_geoms: list[int]) -> np.ndarray:
        """Sum 3-D contact force (world frame) for a foot."""
        force = np.zeros(3, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in foot_geoms or c.geom2 in foot_geoms) and (
                c.geom1 == self._floor_geom or c.geom2 == self._floor_geom
            ):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                force += f6[:3].astype(np.float32)
        return force

    def _foot_in_contact(self, foot_geoms: list[int], threshold: float = 1.0) -> bool:
        return np.linalg.norm(self._foot_contact_force(foot_geoms)) > threshold

    def _foot_vel_xy(self, body_id: int) -> float:
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, body_id)
        vel = jacp @ self._qvel()
        return float(np.linalg.norm(vel[:2]))

    def _qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos).reshape(-1)

    def _qvel(self) -> np.ndarray:
        return np.asarray(self.data.qvel).reshape(-1)

    # ------------------------------------------------------------------
    # Frame transforms
    # ------------------------------------------------------------------

    def _world_to_body_rot(self) -> np.ndarray:
        """3×3 rotation matrix: world → body frame (R^T)."""
        qpos = self._qpos()
        w, x, y, z = qpos[3:7]
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        return R.T.astype(np.float32)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict[str, np.ndarray]:
        R_wb = self._world_to_body_rot()
        qpos = self._qpos()
        qvel = self._qvel()

        # IMU linear acceleration: use last 5ms physics substep to match Isaac Lab ImuCfg (instantaneous).
        # specific_force = a_world - g_world, rotated to body frame.
        a_world   = self._last_physics_acc
        lin_acc_b = (R_wb @ (a_world - np.array([0.0, 0.0, -9.81]))).astype(np.float32)

        # Angular velocity: free-joint qvel[3:6] is in LOCAL frame in MuJoCo.
        ang_vel_b = qvel[3:6].astype(np.float32)

        # Projected gravity (normalised gravity vector in body frame)
        proj_grav = (R_wb @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

        joint_pos_rel = (qpos[self._qpos_ids] - self._default_joint_pos).astype(np.float32)
        joint_vel     = qvel[self._qvel_ids].astype(np.float32)

        gait_obs = np.array([math.sin(self.gait_phase), math.cos(self.gait_phase)], dtype=np.float32)

        # Policy / actor obs — 50 dims (identical concatenation order to Isaac Lab)
        obs_50 = np.concatenate([
            lin_acc_b,         # 3
            ang_vel_b,         # 3
            proj_grav,         # 3
            self.commands,     # 3
            joint_pos_rel,     # 12
            joint_vel,         # 12
            self.prev_actions, # 12
            gait_obs,          # 2
        ])  # total = 50

        if self._history is not None:
            self._history.append(obs_50.copy())
            actor_obs = np.concatenate(list(self._history))
        else:
            actor_obs = obs_50

        # Privileged / critic obs — 10 extra dims
        lin_vel_b = (R_wb @ qvel[:3]).astype(np.float32)
        height    = np.array([qpos[2]], dtype=np.float32)
        r_force   = self._foot_contact_force(self._r_foot_geoms)
        l_force   = self._foot_contact_force(self._l_foot_geoms)
        priv      = np.concatenate([lin_vel_b, height, r_force, l_force])  # 10

        if self.critic_has_privileged_info:
            critic_obs = np.concatenate([obs_50, priv])  # 60
        else:
            critic_obs = obs_50.copy()

        return {"actor": actor_obs.astype(np.float32), "critic": critic_obs.astype(np.float32)}

    # ------------------------------------------------------------------
    # Gait state update
    # ------------------------------------------------------------------

    def _update_gait(self) -> None:
        r = self._foot_in_contact(self._r_foot_geoms)
        l = self._foot_in_contact(self._l_foot_geoms)
        self._in_contact[:] = [r, l]

        # Touchdown / liftoff events (before updating prev)
        self._touchdown[:] = ~self._in_contact_prev & self._in_contact

        for i, (in_c, was_c) in enumerate(zip(self._in_contact, self._in_contact_prev)):
            if in_c:
                self._contact_time[i] += STEP_DT
                self._air_time[i] = 0.0
            else:
                self._air_time[i] += STEP_DT
                self._contact_time[i] = 0.0

            if was_c and not in_c:
                # Liftoff — reset swing tracking
                self._feet_pos_liftoff[i] = self.data.body(
                    "r_foot_link_1" if i == 0 else "l_foot_link_1"
                ).xpos.copy()
                self._knee_bend_max[i] = 0.0

            if not in_c:
                # Track max knee bend during swing
                knee_bend = abs(self._qpos()[self._qpos_ids[KNEE_IDX[i]]])
                if knee_bend > self._knee_bend_max[i]:
                    self._knee_bend_max[i] = knee_bend

        self._in_contact_prev[:] = self._in_contact
        both_up = not r and not l
        self._both_air_steps = (self._both_air_steps + 1) if both_up else 0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> float:
        w = self._reward_w
        if not w:
            return 0.0

        R_wb      = self._world_to_body_rot()
        qpos = self._qpos()
        qvel = self._qvel()
        lin_vel_b = (R_wb @ qvel[:3]).astype(np.float32)
        ang_vel_b = qvel[3:6].astype(np.float32)
        proj_grav = (R_wb @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

        joint_pos = qpos[self._qpos_ids].astype(np.float32)
        joint_vel = qvel[self._qvel_ids].astype(np.float32)
        efforts   = self.data.actuator_force[self._act_ids].astype(np.float32)
        joint_acc = ((joint_vel - self._prev_qvel[self._qvel_ids]) / STEP_DT).astype(np.float32)

        r_foot_pos = self.data.body("r_foot_link_1").xpos.astype(np.float32)
        l_foot_pos = self.data.body("l_foot_link_1").xpos.astype(np.float32)
        feet_pos   = np.stack([r_foot_pos, l_foot_pos])

        r_force = self._foot_contact_force(self._r_foot_geoms)
        l_force = self._foot_contact_force(self._l_foot_geoms)
        in_c    = self._in_contact.astype(np.float32)
        r_vel_xy = self._foot_vel_xy(self._r_foot_body)
        l_vel_xy = self._foot_vel_xy(self._l_foot_body)

        # Torch tensors — all batched as [1, ...]
        cmd_t   = _t1(self.commands)
        lvb_t   = _t1(lin_vel_b)
        avb_t   = _t1(ang_vel_b)
        pg_t    = _t1(proj_grav)
        jp_t    = _t1(joint_pos)
        jv_t    = _t1(joint_vel)
        eff_t   = _t1(efforts)
        act_t   = _t1(action.astype(np.float32))
        pact_t  = _t1(self.prev_actions)
        jacc_t  = _t1(joint_acc)
        def_t   = _t1(self._default_joint_pos)
        lmin_t  = _t1(JLIM_MIN)
        lmax_t  = _t1(JLIM_MAX)
        fp_t    = torch.tensor(feet_pos, dtype=torch.float32).unsqueeze(0)   # [1,2,3]
        fc_t    = _t1(in_c)                                                    # [1,2]
        air_t   = _t1(self._air_time)
        ctt_t   = _t1(self._contact_time)
        td_t    = torch.tensor(self._touchdown.astype(np.float32)).unsqueeze(0)  # [1,2]
        km_t    = torch.tensor(self._knee_bend_max, dtype=torch.float32).unsqueeze(0)  # [1,2]
        bpos_t  = _t1(qpos[:3].astype(np.float32))
        gp_t    = _t0(self.gait_phase)
        fvel_t  = torch.tensor([[r_vel_xy, l_vel_xy]], dtype=torch.float32)   # [1,2]

        def _w(key: str, val: torch.Tensor) -> float:
            wt = w.get(key, 0.0)
            return float(val.squeeze()) * wt if wt != 0.0 else 0.0

        total = 0.0

        # Locomotion
        total += _w("track_lin_vel_xy_exp", R.track_lin_vel_xy_exp(cmd_t, lvb_t, std=0.15))
        total += _w("track_ang_vel_z_exp",  R.track_ang_vel_z_exp(cmd_t, avb_t, std=0.15))

        # Stability
        total += w.get("termination_penalty", 0.0) * float(self._is_terminated())
        total += _w("lin_vel_z_l2",       R.lin_vel_z_l2(lvb_t))
        total += _w("ang_vel_xy_l2",      R.ang_vel_xy_l2(avb_t))
        total += _w("flat_orientation_l2",R.flat_orientation_l2(pg_t))

        # Smoothness
        total += _w("action_rate_l2",  R.action_rate_l2(act_t, pact_t))
        total += _w("dof_torques_l2",  R.joint_torques_l2(eff_t))
        total += _w("dof_acc_l2",      torch.sum(jacc_t ** 2, dim=1))
        total += _w("dof_pos_limits",  R.joint_pos_limits(jp_t, lmin_t, lmax_t))

        # Gait shaping
        total += _w("feet_air_time",   R.feet_air_time_positive_biped(
            air_t, ctt_t, cmd_t, threshold=0.2, min_speed_command_threshold=0.05))
        total += _w("feet_slide",      R.feet_slide_with_vel(fc_t, fvel_t))
        total += _w("swing_foot_height", R.swing_foot_height(
            fp_t, fc_t.bool(), min_height=0.02, max_height=0.04))
        total += _w("gait_phase_contact", R.gait_phase_contact(
            gp_t, fc_t, cmd_t, min_speed=0.05))
        total += _w("knee_bend_touchdown", R.knee_bend_on_touchdown(
            td_t, km_t, min_bend=0.2, max_bend=1.0))

        # Joint quality
        def _dev(idx):
            return R.joint_deviation_l1(jp_t[:, idx], def_t[:, idx])

        total += _w("joint_deviation_hip",          _dev(HIP_IDX))
        total += _w("joint_deviation_ankle_roll",   _dev(ANKLE_ROLL_IDX))
        total += _w("joint_deviation_ankle_pitch",  _dev(ANKLE_PITCH_IDX))
        total += _w("joint_deviation_knee",         _dev(KNEE_IDX))
        total += _w("ankle_torques_l2",             R.joint_torques_l2(eff_t[:, ANKLE_PITCH_IDX]))
        total += _w("torso_centering",              R.torso_centering_reward(bpos_t, fp_t))

        return total * self._reward_scale

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_terminated(self) -> bool:
        R_wb      = self._world_to_body_rot()
        proj_grav = (R_wb @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
        tilt      = math.acos(float(np.clip(-proj_grav[2], -1.0, 1.0)))
        if tilt > 0.784:
            return True
        if self._both_air_steps > int(1.0 / STEP_DT):
            return True
        return False

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _sample_commands(self) -> None:
        try:
            ranges = self.config.commands["base_velocity"]["ranges"]
            xr = tuple(ranges.get("lin_vel_x", _DEFAULT_CMD_RANGES["lin_vel_x"]))
            yr = tuple(ranges.get("lin_vel_y", _DEFAULT_CMD_RANGES["lin_vel_y"]))
            wr = tuple(ranges.get("ang_vel_z", _DEFAULT_CMD_RANGES["ang_vel_z"]))
        except (AttributeError, KeyError, TypeError):
            xr = _DEFAULT_CMD_RANGES["lin_vel_x"]
            yr = _DEFAULT_CMD_RANGES["lin_vel_y"]
            wr = _DEFAULT_CMD_RANGES["ang_vel_z"]
        self.commands[:] = [
            np.random.uniform(*xr),
            np.random.uniform(*yr),
            np.random.uniform(*wr),
        ]

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Override the velocity command (useful for manual testing)."""
        self.commands[:] = [vx, vy, wz]

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._randomizer.randomize()
        mujoco.mj_forward(self.model, self.data)

        # Reset EMA filter to current joint positions (matches biped_env.py _reset_idx)
        self.filtered_targets[:] = self._qpos()[self._qpos_ids].copy()
        self.prev_actions[:] = 0.0
        self._prev_qvel[:] = self._qvel().copy()

        # Random gait phase at reset (matches Isaac Lab)
        self.gait_phase = np.random.uniform(0.0, 2.0 * math.pi)

        # Reset gait tracking
        self._air_time[:]        = 0.0
        self._contact_time[:]    = 0.0
        self._in_contact[:]      = False
        self._in_contact_prev[:] = True  # pretend both feet were on ground
        self._knee_bend_max[:]   = 0.0
        self._touchdown[:]       = False
        self._both_air_steps     = 0

        # Reset history
        if self._history is not None:
            for _ in range(self.history_size):
                self._history.append(np.zeros(self._base_actor_dim, dtype=np.float32))

        self._sample_commands()

        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # [-1,1] → joint position targets (same formula as Isaac Lab _pre_physics_step)
        norm_01 = (action + 1.0) * 0.5
        targets = JLIM_MIN + norm_01 * JRANGE

        # EMA action filter (matches biped_env_cfg.py action_filter_alpha=0.4)
        self.filtered_targets[:] = (
            ACTION_FILTER_ALPHA * targets + (1.0 - ACTION_FILTER_ALPHA) * self.filtered_targets
        )

        # Cache velocity before stepping (for reward joint_acc computation on line 380)
        self._prev_qvel[:] = self._qvel().copy()

        # Advance gait phase (matches _get_observations in biped_env.py)
        v_cmd = float(np.linalg.norm(self.commands[:2]))
        self.gait_phase = (
            self.gait_phase + STEP_DT * 2.0 * math.pi * GAIT_CLOCK_FREQ * v_cmd
        ) % (2.0 * math.pi)

        # Python PD torque — matches Isaac Lab ImplicitActuatorCfg:
        #   tau = clip(kp*(q_des - q) + kd*(0 - qdot), -limit, limit)
        # kd is INSIDE the clip, same as Isaac Lab. Passive joint damping set to 0 in XML.
        pos_err = self.filtered_targets - self._qpos()[self._qpos_ids]
        tau = np.clip(80.0 * pos_err - 8.0 * self._qvel()[self._qvel_ids], -3.0, 3.0)
        self.data.ctrl[self._act_ids] = tau

        # Physics: DECIMATION sub-steps at 200 Hz
        # Run all but last substep, then capture last-substep velocity for instantaneous IMU acc
        for _ in range(DECIMATION - 1):
            mujoco.mj_step(self.model, self.data)
        _vel_before_last = self.data.qvel[:3].copy()
        mujoco.mj_step(self.model, self.data)
        self._last_physics_acc = (self.data.qvel[:3] - _vel_before_last) / PHYSICS_DT
        # Clamp joint velocities to match Isaac Lab velocity_limit=5 rad/s
        self.data.qvel[self._qvel_ids] = np.clip(self.data.qvel[self._qvel_ids], -5.0, 5.0)

        self._update_gait()

        reward     = self._compute_reward(action)
        terminated = self._is_terminated()

        obs = self._build_obs()
        self.prev_actions[:] = action.copy()

        return obs, reward, terminated, False, {}
