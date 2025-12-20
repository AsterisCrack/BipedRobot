import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
import mujoco
from envs.base_env import BaseEnv, FeetContactBuffer
from envs.utils.randomizer import Randomizer
from envs.rewards.walk_reward import WalkReward
from envs.rewards.target_reward import TargetReward
from envs.utils.mirroring import mirror_observation, mirror_action
from utils import NoConfig

class MujocoEnv(BaseEnv):
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
        
        # Mujoco IDs
        self.l_feet_body = self.model.body('l_foot').id
        self.r_feet_body = self.model.body('r_foot').id
        self.l_feet_geom = self.model.geom('l_foot').id
        self.r_feet_geom = self.model.geom('r_foot').id
        self.floor_geom = self.model.geom('floor').id
        
        # State variables
        self.target = 0.0
        self.steps_in_target = 0
        self.prev_joint_pos = np.zeros(12)
        self.prev_actions = np.zeros(12)
        self.feet_contact_buffer = FeetContactBuffer(max_time=0.2)
        self.last_step_time = 0.0
        self.prev_left_contact = False
        self.prev_right_contact = False
        
        # Randomizer
        # randomization_cfg is passed via env_config usually, but schema has it top level in Config. 
        # For simplicity, we assume randomize_* are passed in some form or accessed via config object.
        # Following TrainConfig structure:
        self.randomizer = Randomizer(self.model, random_config)
        
        # Reward
        objective = getattr(self.config, "objective", "walk")
        reward_weights = getattr(self.config, "reward_weights", {})
        
        if objective == "target":
            self.reward_provider = TargetReward(self, reward_weights)
        else:
            self.reward_provider = WalkReward(self, reward_weights)
            
        # Spaces
        self._setup_spaces()
        
        self.reset(seed=seed)

    def _setup_spaces(self):
        nq, nv, nu = self.model.nq, self.model.nv, self.model.nu
        self.normal_obs_dim = nq + nv
        # qpos (nq), qvel (nv), feet_forces (6), floor_friction (3), damping (nv), mass (nbody)
        self.privileged_obs_dim = nq + nv + 6 + 3 + nv + self.model.nbody
        
        objective = getattr(self.config, "objective", "walk")
        if objective == "target":
            self.normal_obs_dim += 1
            self.privileged_obs_dim += 1

        self.action_space = Box(low=-np.pi, high=np.pi, shape=(nu,), dtype=np.float32)
        
        def get_dim(obs_type):
            return self.normal_obs_dim if obs_type == "normal" else self.privileged_obs_dim

        actor_dim = get_dim(self.actor_obs_type)
        critic_dim = get_dim(self.critic_obs_type)
        
        if self.history_size > 0:
            actor_dim = self.history_size * (actor_dim + nu)
            critic_dim = self.history_size * (critic_dim + nu)
            self.actor_history = np.zeros((self.history_size, actor_dim // self.history_size), dtype=np.float32)
            self.critic_history = np.zeros((self.history_size, critic_dim // self.history_size), dtype=np.float32)

        self.observation_space = Dict({
            "actor": Box(low=-np.inf, high=np.inf, shape=(actor_dim,), dtype=np.float32),
            "critic": Box(low=-np.inf, high=np.inf, shape=(critic_dim,), dtype=np.float32)
        })

    def _get_privileged_obs(self):
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        
        # Forces
        l_force, r_force = np.zeros(3), np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.l_feet_geom or contact.geom2 == self.l_feet_geom:
                f = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, f); l_force += f[:3]
            if contact.geom1 == self.r_feet_geom or contact.geom2 == self.r_feet_geom:
                f = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, f); r_force += f[:3]
        
        obs = np.concatenate([
            qpos, qvel, l_force, r_force,
            self.model.geom_friction[self.floor_geom],
            self.model.dof_damping,
            self.model.body_mass
        ])
        
        objective = getattr(self.config, "objective", "walk")
        if objective == "target":
            obs = np.append(obs, np.abs(self.data.qpos[3] - self.target))
            
        return obs.astype(np.float32)

    def _get_normal_obs(self):
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        imu_noise, vel_noise = self.randomizer.get_noise()
        qpos[3:7] += np.random.normal(0, imu_noise, size=4)
        qvel[0:3] += np.random.normal(0, vel_noise, size=3)
        
        obs = np.concatenate([qpos, qvel])
        objective = getattr(self.config, "objective", "walk")
        if objective == "target":
            obs = np.append(obs, np.abs(self.data.qpos[3] - self.target))
            
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
             self._np_random, seed = gym.utils.seeding.np_random(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.randomizer.randomize()
        mujoco.mj_forward(self.model, self.data)
        
        self.prev_joint_pos = self.data.qpos[7:].copy()
        self.prev_actions = self.data.ctrl.copy()
        self.feet_contact_buffer.clear()
        self.last_step_time = self.data.time
        
        objective = getattr(self.config, "objective", "walk")
        if objective == "target":
            self._reset_target()

        obs = self._build_obs(np.zeros(self.model.nu))
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        reward = self.reward_provider.compute(action)
        terminated = self._is_terminated()
        
        obs = self._build_obs(action)
        self.prev_actions = action.copy()
        
        return obs, reward, terminated, False, self.reward_provider.get_reward_info()

    def _build_obs(self, action):
        normal = self._get_normal_obs()
        priv = self._get_privileged_obs()
        
        def process(obs_type, obs_val):
            if self.history_size > 0:
                hist = self.actor_history if obs_type == "actor" else self.critic_history
                # Simplified history logic: [obs, action]
                combined = np.concatenate([obs_val, action])
                hist[:] = np.roll(hist, -1, axis=0)
                hist[-1] = combined
                return hist.flatten()
            return obs_val

        actor_obs = process("actor", normal if self.actor_obs_type == "normal" else priv)
        critic_obs = process("critic", normal if self.critic_obs_type == "normal" else priv)
        
        return {"actor": actor_obs, "critic": critic_obs}

    def _reset_target(self):
        self.target = np.random.uniform(0, 2 * np.pi)

    def _check_reached_target(self):
        yaw_diff = np.abs(self.data.qpos[3] - self.target)
        if yaw_diff < 0.1:
            self.steps_in_target += 1
            if self.steps_in_target >= 100:
                self.steps_in_target = 0
                self._reset_target()
                return True
        else:
            self.steps_in_target = 0
        return False

    def _get_body_orientation(self, body_id):
        return self.data.xquat[body_id]
