"""MuJoCo reward - identical terms to BipedEnv._get_rewards().

Imports the shared @torch.jit.script functions from envs/isaaclab/rewards/rewards.py
and calls them by converting MuJoCo numpy state to single-batch (N=1) torch tensors.
No isaac lab dependency required - rewards.py is pure torch after the quat_apply import
was removed.
"""

import numpy as np
import torch
from envs.isaaclab.rewards import rewards as rwd

# Joint names for per-group deviation rewards.
_HIP_JOINT_NAMES        = ["r_hip_z", "r_hip_x", "l_hip_z", "l_hip_x"]
_ANKLE_ROLL_JOINT_NAMES = ["r_ankle_x", "l_ankle_x"]
_KNEE_JOINT_NAMES       = ["r_knee", "l_knee"]

_DEFAULT_WEIGHTS: dict[str, float] = {
    "track_lin_vel_xy_exp":        2.0,
    "track_ang_vel_z_exp":         1.0,
    "termination_penalty":        -10.0,
    "lin_vel_z_l2":               -0.1,
    "ang_vel_xy_l2":              -0.05,
    "flat_orientation_l2":        -2.0,
    "action_rate_l2":             -0.01,
    "dof_torques_l2":             -2.0e-3,
    "dof_acc_l2":                 -1.0e-6,
    "dof_pos_limits":             -1.0,
    "feet_air_time":               1.0,
    "feet_slide":                 -0.1,
    "joint_deviation_hip":        -0.2,
    "joint_deviation_ankle_roll": -0.2,
    "step_length":                 0.0,
    "swing_foot_height":           0.0,
    "torso_centering":             0.0,
    "knee_bend_touchdown":         0.0,
}


def _t(arr) -> torch.Tensor:
    """numpy array → float32 torch tensor with leading batch dim (1, ...)."""
    return torch.as_tensor(np.asarray(arr, dtype=np.float32)).unsqueeze(0)


class MujocoReward:
    """Single-environment reward that mirrors BipedEnv._get_rewards().

    Maintains the same per-foot contact/stride/knee-bend buffers that Isaac Lab's
    BipedEnv keeps in GPU tensors, but as plain numpy arrays for the CPU env.

    Feet order is [right, left] throughout, matching Isaac Lab's feet_indices.
    """

    def __init__(self, env, weights: dict | None = None):
        self.env = env
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.reward_dict: dict[str, float] = {}

        # Resolve joint qpos indices (offset past the 7-DOF free joint).
        def _qidx(name: str) -> int:
            return env.model.joint(name).qposadr - 7

        self._hip_idx   = [_qidx(n) for n in _HIP_JOINT_NAMES]
        self._ankle_idx = [_qidx(n) for n in _ANKLE_ROLL_JOINT_NAMES]
        self._knee_idx  = [_qidx(n) for n in _KNEE_JOINT_NAMES]

        # Joint position limits from model (rows 1..njnt, skipping the free joint at row 0).
        jnt_range = env.model.jnt_range[1:]       # (12, 2)
        self._jnt_min = _t(jnt_range[:, 0])       # (1, 12)
        self._jnt_max = _t(jnt_range[:, 1])       # (1, 12)
        self._default_jpos = _t(env.default_joint_pos)  # (1, 12)

        # Per-foot buffers - [right, left] order.
        self._air_time      = np.zeros(2, dtype=np.float32)
        self._contact_time  = np.zeros(2, dtype=np.float32)
        self._in_contact    = np.ones(2, dtype=bool)   # start in contact
        self._pos_liftoff   = np.zeros((2, 3), dtype=np.float32)
        self._knee_bend_max = np.zeros(2, dtype=np.float32)
        # Last foot to touch down: 0=right, 1=left. Init to 1 so right is valid first.
        self._last_foot = 1

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._air_time[:]      = 0.0
        self._contact_time[:]  = 0.0
        self._in_contact[:]    = True
        self._pos_liftoff[:]   = 0.0
        self._knee_bend_max[:] = 0.0
        self._last_foot        = 1
        self.reward_dict       = {}

    def get_reward_info(self) -> dict[str, float]:
        return self.reward_dict

    # ------------------------------------------------------------------
    def compute(self, action: np.ndarray) -> float:
        env  = self.env
        data = env.data
        dt   = env.dt

        # ---- Body-frame kinematics ------------------------------------------
        R_inv       = env._get_body_rot_inv()
        lin_vel_b   = _t(R_inv @ data.qvel[0:3])           # (1, 3)
        ang_vel_b   = _t(R_inv @ data.qvel[3:6])           # (1, 3)
        proj_grav_b = _t(R_inv @ np.array([0., 0., -1.]))  # (1, 3)
        commands    = _t(env.commands)                      # (1, 3)

        # ---- Joint quantities -----------------------------------------------
        actions_t  = _t(action)               # (1, 12)
        prev_act_t = _t(env.prev_actions)     # (1, 12)
        joint_pos  = _t(data.qpos[7:])        # (1, 12)
        joint_eff  = _t(data.actuator_force)  # (1, 12)
        joint_acc  = _t(data.qacc[6:])        # (1, 12)

        # ---- Feet state: [right, left] --------------------------------------
        r_pos_w    = data.geom_xpos[env.r_feet_geom].copy()   # (3,)
        l_pos_w    = data.geom_xpos[env.l_feet_geom].copy()   # (3,)
        feet_pos_np = np.stack([r_pos_w, l_pos_w])             # (2, 3)
        feet_pos    = torch.tensor(feet_pos_np, dtype=torch.float32).unsqueeze(0)  # (1, 2, 3)
        base_pos    = _t(data.qpos[:3])                        # (1, 3)

        r_con = env._feet_in_contact(env.r_feet_geom)
        l_con = env._feet_in_contact(env.l_feet_geom)
        in_contact      = np.array([r_con, l_con])
        feet_in_contact = torch.tensor([[r_con, l_con]])       # (1, 2)

        # ---- Liftoff / touchdown events -------------------------------------
        liftoff   = self._in_contact & ~in_contact
        touchdown = ~self._in_contact & in_contact

        for i in range(2):
            if liftoff[i]:
                self._pos_liftoff[i]   = feet_pos_np[i]
                self._knee_bend_max[i] = 0.0

        stride_np   = np.linalg.norm(feet_pos_np - self._pos_liftoff, axis=-1)  # (2,)
        stride_dist = torch.tensor(stride_np, dtype=torch.float32).unsqueeze(0)  # (1, 2)

        if touchdown[0]:
            self._last_foot = 0
        if touchdown[1]:
            self._last_foot = 1

        # ---- Air / contact time update --------------------------------------
        for i in range(2):
            if in_contact[i]:
                self._contact_time[i] += dt
                self._air_time[i]      = 0.0
            else:
                self._air_time[i]     += dt
                self._contact_time[i]  = 0.0

        # ---- Knee bend (track max during swing, reset on touchdown) ---------
        knee_pos     = data.qpos[7:][self._knee_idx]
        knee_default = env.default_joint_pos[self._knee_idx]
        knee_bend    = np.abs(knee_pos - knee_default)
        self._knee_bend_max = np.where(
            ~in_contact,
            np.maximum(self._knee_bend_max, knee_bend),
            self._knee_bend_max,
        )

        air_t       = torch.tensor(self._air_time,      dtype=torch.float32).unsqueeze(0)  # (1,2)
        contact_t   = torch.tensor(self._contact_time,  dtype=torch.float32).unsqueeze(0)  # (1,2)
        touchdown_t = torch.tensor(touchdown).unsqueeze(0)                                  # (1,2)
        knee_max_t  = torch.tensor(self._knee_bend_max, dtype=torch.float32).unsqueeze(0)  # (1,2)

        # ====================================================================
        # Reward terms - same as BipedEnv._get_rewards()
        # ====================================================================

        r_track_lin  = rwd.track_lin_vel_xy_exp(commands, lin_vel_b, std=0.25).item()
        r_track_ang  = rwd.track_ang_vel_z_exp(commands, ang_vel_b, std=0.25).item()
        r_term       = float(env._is_terminated())
        r_lin_vel_z  = rwd.lin_vel_z_l2(lin_vel_b).item()
        r_ang_vel_xy = rwd.ang_vel_xy_l2(ang_vel_b).item()
        r_flat_ori   = rwd.flat_orientation_l2(proj_grav_b).item()
        r_act_rate   = rwd.action_rate_l2(actions_t, prev_act_t).item()
        r_torques    = rwd.joint_torques_l2(joint_eff).item()
        r_dof_acc    = torch.sum(torch.square(joint_acc)).item()
        r_pos_lim    = rwd.joint_pos_limits(joint_pos, self._jnt_min, self._jnt_max).item()

        r_air_time = rwd.feet_air_time_positive_biped(
            air_t, contact_t, commands, threshold=0.5, min_speed_command_threshold=0.05
        ).item()
        # Only when the expected-air foot is actually airborne (alternating gait filter).
        if in_contact[(self._last_foot + 1) % 2]:
            r_air_time = 0.0

        r_feet_slide = rwd.feet_slide_with_vel(
            feet_in_contact.float(),
            _t([env._get_foot_vel_xy_norm(env.r_feet_body),
                env._get_foot_vel_xy_norm(env.l_feet_body)]),
        ).item()

        r_dev_hip   = rwd.joint_deviation_l1(
            joint_pos[:, self._hip_idx], self._default_jpos[:, self._hip_idx]
        ).item()
        r_dev_ankle = rwd.joint_deviation_l1(
            joint_pos[:, self._ankle_idx], self._default_jpos[:, self._ankle_idx]
        ).item()

        r_step_len  = rwd.step_length(
            touchdown_t, stride_dist, commands, min_speed_command_threshold=0.1
        ).item()
        r_swing_h   = rwd.swing_foot_height(
            feet_pos, feet_in_contact, min_height=0.05, max_height=0.15
        ).item()
        r_torso     = rwd.torso_centering_reward(base_pos, feet_pos).item()
        r_knee      = rwd.knee_bend_on_touchdown(
            touchdown_t, knee_max_t, min_bend=0.2, max_bend=1.0
        ).item()

        # Reset knee-bend max on touchdown (after using the value above).
        for i in range(2):
            if touchdown[i]:
                self._knee_bend_max[i] = 0.0

        self._in_contact = in_contact.copy()

        # ---- Weighted sum ---------------------------------------------------
        terms = {
            "track_lin_vel_xy_exp":        r_track_lin,
            "track_ang_vel_z_exp":         r_track_ang,
            "termination_penalty":         r_term,
            "lin_vel_z_l2":               r_lin_vel_z,
            "ang_vel_xy_l2":              r_ang_vel_xy,
            "flat_orientation_l2":        r_flat_ori,
            "action_rate_l2":             r_act_rate,
            "dof_torques_l2":             r_torques,
            "dof_acc_l2":                 r_dof_acc,
            "dof_pos_limits":             r_pos_lim,
            "feet_air_time":              r_air_time,
            "feet_slide":                 r_feet_slide,
            "joint_deviation_hip":        r_dev_hip,
            "joint_deviation_ankle_roll": r_dev_ankle,
            "step_length":                r_step_len,
            "swing_foot_height":          r_swing_h,
            "torso_centering":            r_torso,
            "knee_bend_touchdown":        r_knee,
        }

        total = sum(v * self.weights.get(k, 0.0) for k, v in terms.items())
        self.reward_dict = {
            f"reward_{k}": v * self.weights.get(k, 0.0) for k, v in terms.items()
        }
        return total
