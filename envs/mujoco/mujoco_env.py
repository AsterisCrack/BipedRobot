import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
import mujoco
from envs.mujoco.base_env import BaseEnv
from envs.utils.randomizer import Randomizer
from envs.rewards.mujoco_reward import MujocoReward
from envs.utils.mirroring import mirror_observation, mirror_action
from utils import NoConfig

# Default velocity command ranges (matching Isaac Lab defaults).
_DEFAULT_CMD_RANGES = {
    "lin_vel_x": (-0.5, 1.0),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.5, 0.5),
}


class MujocoEnv(BaseEnv):
    """Biped MuJoCo environment - mirrors Isaac Lab's BipedEnv.

    Observation space (48-dim actor obs, identical layout to BipedEnv):
        [lin_acc (3), ang_vel_b (3), projected_gravity (3),
         commands (3), joint_pos_rel (12), joint_vel (12), prev_actions (12)]

    Privileged critic obs:
        actor obs + feet_forces (6) + floor_friction (3) + dof_damping (nv) + body_mass (nbody)

    Action space: normalized [-1, 1] mapped to joint control range (same as Isaac Lab).

    Reward: identical terms to BipedEnv._get_rewards() via shared reward functions.
    Termination: tilt angle > ~45° OR base height < 0.15 m (matches Isaac Lab).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, history_size=0, sim_frequency=100,
                 random_config=None, seed=None, actor_obs="normal", critic_obs="privileged",
                 env_config=None):

        self.name = "MujocoEnv"
        xml_path = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"
        super().__init__(xml_path, render_mode)

        self.config = env_config or NoConfig()
        self.history_size = history_size
        self.actor_obs_type = actor_obs
        self.critic_obs_type = critic_obs

        # Simulation frequency
        self.model.opt.timestep = 1.0 / sim_frequency
        self.dt = self.model.opt.timestep

        # MuJoCo body / geom IDs
        self.l_feet_body = self.model.body("l_foot").id
        self.r_feet_body = self.model.body("r_foot").id
        self.l_feet_geom = self.model.geom("l_foot").id
        self.r_feet_geom = self.model.geom("r_foot").id
        self.floor_geom  = self.model.geom("floor").id

        # Action mapping: normalized [-1, 1] → actuator control range
        self.ctrl_min   = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_max   = self.model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_range = self.ctrl_max - self.ctrl_min

        # Default joint positions (neutral pose, used for relative obs and deviation rewards)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.default_joint_pos = self.data.qpos[7:].copy()  # 12 joints

        # Joint group indices into qpos[7:] (for reward computation)
        def _qidx(name: str) -> int:
            return self.model.joint(name).qposadr - 7

        self.hip_indices        = [_qidx(n) for n in ["r_hip_z", "r_hip_x", "l_hip_z", "l_hip_x"]]
        self.knee_indices       = [_qidx(n) for n in ["r_knee",  "l_knee"]]
        self.ankle_roll_indices = [_qidx(n) for n in ["r_ankle_x", "l_ankle_x"]]

        # State
        self.commands      = np.zeros(3)              # [vx, vy, wz] velocity commands
        self.prev_actions  = np.zeros(self.model.nu)
        self.prev_lin_vel  = np.zeros(3)

        # Mirroring
        self.enable_mirroring           = getattr(self.config, "enable_mirroring", False)
        self.use_mirroring_this_episode = False

        # Randomizer
        self.randomizer = Randomizer(self.model, random_config)

        # Reward provider
        reward_weights = getattr(self.config, "reward_weights", {})
        self.reward_provider = MujocoReward(self, reward_weights)

        # Spaces
        self._setup_spaces()
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Space setup
    # ------------------------------------------------------------------

    def _setup_spaces(self):
        nu = self.model.nu   # 12 joints
        nv = self.model.nv   # DOF count (6 free + 12 joints = 18)

        # Actor obs: 48 dims (matches Isaac Lab proprioceptive obs)
        self.actor_obs_dim = 48
        # Privileged extras: feet forces (6) + floor friction (3) + dof damping (nv) + body mass (nbody)
        priv_extras_dim    = 6 + 3 + nv + self.model.nbody
        self.critic_obs_dim = self.actor_obs_dim + priv_extras_dim

        self.action_space = Box(low=-1.0, high=1.0, shape=(nu,), dtype=np.float32)

        actor_dim  = self.actor_obs_dim
        critic_dim = self.actor_obs_dim if self.critic_obs_type == "normal" else self.critic_obs_dim

        if self.history_size > 0:
            step_dim_actor  = actor_dim  + nu
            step_dim_critic = critic_dim + nu
            actor_dim  = self.history_size * step_dim_actor
            critic_dim = self.history_size * step_dim_critic
            self.actor_history  = np.zeros((self.history_size, step_dim_actor),  dtype=np.float32)
            self.critic_history = np.zeros((self.history_size, step_dim_critic), dtype=np.float32)

        self.observation_space = Dict({
            "actor":  Box(low=-np.inf, high=np.inf, shape=(actor_dim,),  dtype=np.float32),
            "critic": Box(low=-np.inf, high=np.inf, shape=(critic_dim,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_body_rot_inv(self) -> np.ndarray:
        """Rotation matrix world → body frame (R^T)."""
        w, x, y, z = self.data.qpos[3:7]
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
        return R.T

    def _get_body_orientation(self, body_id: int) -> np.ndarray:
        return self.data.xquat[body_id]

    def _feet_in_contact(self, geom_id: int, force_threshold: float = 1.0) -> bool:
        """Return True if foot geom has a contact with force > threshold (N)."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                if np.linalg.norm(f[:3]) > force_threshold:
                    return True
        return False

    def _get_foot_vel_xy_norm(self, body_id: int) -> float:
        """XY-plane speed of a body (world frame) via body Jacobian."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, body_id)
        vel_world = jacp @ self.data.qvel
        return float(np.linalg.norm(vel_world[:2]))

    def _is_terminated(self) -> bool:
        """Terminate on excessive tilt (matching Isaac Lab) or critically low height."""
        proj_gravity = self._get_body_rot_inv() @ np.array([0.0, 0.0, -1.0])
        tilt_angle   = np.arccos(np.clip(-proj_gravity[2], -1.0, 1.0))
        return bool(tilt_angle > 0.784 or self.data.qpos[2] < 0.15)

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _get_actor_obs(self) -> np.ndarray:
        """48-dim proprioceptive obs - identical layout to BipedEnv."""
        R_inv = self._get_body_rot_inv()

        lin_acc        = R_inv @ ((self.data.qvel[0:3] - self.prev_lin_vel) / self.dt)
        ang_vel_b      = R_inv @ self.data.qvel[3:6]
        projected_grav = R_inv @ np.array([0.0, 0.0, -1.0])

        joint_pos_rel = self.data.qpos[7:] - self.default_joint_pos  # 12
        joint_vel     = self.data.qvel[6:]                            # 12

        obs = np.concatenate([
            lin_acc,            # 3
            ang_vel_b,          # 3
            projected_grav,     # 3
            self.commands,      # 3
            joint_pos_rel,      # 12
            joint_vel,          # 12
            self.prev_actions,  # 12
        ])
        return obs.astype(np.float32)

    def _get_privileged_obs(self) -> np.ndarray:
        """Actor obs + privileged extras (richer than Isaac Lab's critic obs by design)."""
        actor = self._get_actor_obs()

        l_force, r_force = np.zeros(3), np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.l_feet_geom or contact.geom2 == self.l_feet_geom:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                l_force += f[:3]
            if contact.geom1 == self.r_feet_geom or contact.geom2 == self.r_feet_geom:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                r_force += f[:3]

        obs = np.concatenate([
            actor,
            l_force, r_force,                           # 6
            self.model.geom_friction[self.floor_geom],  # 3
            self.model.dof_damping,                     # nv
            self.model.body_mass,                       # nbody
        ])
        return obs.astype(np.float32)

    def _build_obs(self, action: np.ndarray) -> dict:
        actor_obs  = self._get_actor_obs()
        critic_obs = actor_obs if self.critic_obs_type == "normal" else self._get_privileged_obs()

        def process(obs_val, history_buf):
            if self.history_size > 0:
                combined = np.concatenate([obs_val, action])
                history_buf[:] = np.roll(history_buf, -1, axis=0)
                history_buf[-1] = combined
                return history_buf.flatten()
            return obs_val

        return {
            "actor":  process(actor_obs,  self.actor_history  if self.history_size > 0 else None),
            "critic": process(critic_obs, self.critic_history if self.history_size > 0 else None),
        }

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def _sample_commands(self):
        """Sample velocity commands from config ranges (or defaults matching Isaac Lab)."""
        try:
            ranges  = self.config.commands["base_velocity"]["ranges"]
            x_range = ranges.get("lin_vel_x", _DEFAULT_CMD_RANGES["lin_vel_x"])
            y_range = ranges.get("lin_vel_y", _DEFAULT_CMD_RANGES["lin_vel_y"])
            w_range = ranges.get("ang_vel_z", _DEFAULT_CMD_RANGES["ang_vel_z"])
        except (AttributeError, KeyError, TypeError):
            x_range = _DEFAULT_CMD_RANGES["lin_vel_x"]
            y_range = _DEFAULT_CMD_RANGES["lin_vel_y"]
            w_range = _DEFAULT_CMD_RANGES["ang_vel_z"]

        self.commands[0] = np.random.uniform(*x_range)
        self.commands[1] = np.random.uniform(*y_range)
        self.commands[2] = np.random.uniform(*w_range)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.randomizer.randomize()
        mujoco.mj_forward(self.model, self.data)

        self.prev_actions[:] = 0.0
        self.prev_lin_vel    = self.data.qvel[0:3].copy()

        if self.history_size > 0:
            self.actor_history[:]  = 0.0
            self.critic_history[:] = 0.0

        self._sample_commands()
        self.reward_provider.reset()

        if self.enable_mirroring:
            self.use_mirroring_this_episode = np.random.rand() < 0.5
        else:
            self.use_mirroring_this_episode = False

        obs = self._build_obs(np.zeros(self.model.nu))
        if self.use_mirroring_this_episode:
            obs = self._mirror_observation(obs)

        return obs, {}

    def step(self, action: np.ndarray):
        actual_action = action
        if self.use_mirroring_this_episode:
            actual_action = mirror_action(action, self.model.nu)

        actual_action = np.clip(actual_action, -1.0, 1.0)

        # Map normalized action to actuator control range (same as Isaac Lab)
        action_01 = (actual_action + 1.0) * 0.5
        ctrl      = self.ctrl_min + action_01 * self.ctrl_range

        self.prev_lin_vel  = self.data.qvel[0:3].copy()
        self.data.ctrl[:]  = ctrl
        mujoco.mj_step(self.model, self.data)

        reward     = self.reward_provider.compute(actual_action)
        terminated = self._is_terminated()

        obs = self._build_obs(actual_action)
        self.prev_actions = actual_action.copy()

        if self.use_mirroring_this_episode:
            obs = self._mirror_observation(obs)

        return obs, reward, terminated, False, self.reward_provider.get_reward_info()

    # ------------------------------------------------------------------
    # Mirroring
    # ------------------------------------------------------------------

    def _mirror_observation(self, obs_dict: dict) -> dict:
        if self.history_size > 0:
            mirrored = {}
            for key, obs in obs_dict.items():
                obs_dim  = self.actor_obs_dim if (key == "actor" or self.critic_obs_type == "normal") else self.critic_obs_dim
                nu       = self.model.nu
                step_size = obs_dim + nu
                arr = obs.copy()
                for i in range(self.history_size):
                    s = i * step_size
                    arr[s:s + obs_dim]            = mirror_observation(obs[s:s + obs_dim], nu)
                    arr[s + obs_dim:s + step_size] = mirror_action(obs[s + obs_dim:s + step_size], nu)
                mirrored[key] = arr
            return mirrored
        return {k: mirror_observation(v, self.model.nu) for k, v in obs_dict.items()}
