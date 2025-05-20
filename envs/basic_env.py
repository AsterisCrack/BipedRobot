import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import mujoco
from mujoco.glfw import glfw
from utils import free_camera_movement
import scipy.spatial.transform

# Buffer to store timesteps and if there is a feet contacting the ground
# If any stored value has lived longer than max_time, then remove it
class FeetContactBuffer:
    def __init__(self, max_time):
        self.max_time = max_time
        self.buffer = []
        self.time = 0
    
    def add(self,time):
        self.buffer.append(time)
        
    
    def _remove_old_values(self):
        self.buffer = [time for time in self.buffer if self.time - time <= self.max_time]
        self.buffer.sort()
        
    def get(self, time):
        self.time = time
        self._remove_old_values()
        if len(self.buffer) == 0:
            return 0
        return self.buffer[0]  # Return the first value in the buffer, which is the oldest
    
    def clear(self):
        self.buffer = []
        
class BasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, short_history_size=0, long_history_size=0, sim_frequency=0.33, randomize_dynamics=False, randomize_sensors=False, randomize_perturbations=False, random_config=None, seed=None):
        
        super().__init__()
        # Path to robot XML
        xml_path = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.max_episode_steps = 2000  # Large number of steps
        self.l_feet_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot")
        self.r_feet_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot")
        self.name = "BasicEnv"
        
        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        self.randomize_dynamics = randomize_dynamics
        self.randomize_sensors = randomize_sensors
        self.randomize_perturbations = randomize_perturbations
        self.random_config = random_config
        if (randomize_dynamics or randomize_sensors or randomize_perturbations) and random_config is None:
            raise ValueError("random_config must be provided if randomize_dynamics, randomize_sensors or randomize_perturbations are True.")
        self.t_last_perturbation = 0
        self.t_next_perturbation = 0
        
        # Set the simulation frequency
        self.model.opt.timestep = 1.0 / sim_frequency

        # Observation space: Full state (joint positions and velocities)
        obs_dim = self.model.nq + self.model.nv + \
            short_history_size * (self.model.nq + self.model.nv + self.model.nu) + \
            long_history_size * (self.model.nq + self.model.nv + self.model.nu)
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # Action space: Control position of servos
        action_dim = self.model.nu
        self.action_space = Box(low=-np.pi, high=np.pi, shape=(action_dim,), dtype=np.float32)
        
        # Start history and long history
        self.short_history_size = short_history_size
        self.long_history_size = long_history_size
        self.short_history = np.zeros((self.short_history_size, self.model.nq+self.model.nv+action_dim), dtype=np.float32)
        self.long_history = np.zeros((self.long_history_size, self.model.nq+self.model.nv+action_dim), dtype=np.float32)

        # Rendering attributes
        self.window = None
        self.context = None
        self.viewport_width = 1200
        self.viewport_height = 900
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.free_camera = free_camera_movement.FreeCameraMovement(self.model, self.cam, self.scene)
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._initialize_renderer()
            
        # Initialize variables for differentiation
        self.prev_joint_pos = np.zeros(12)
        self.prev_actions = np.zeros(12)
        self.original_height = 0
        self.feet_contact_buffer = FeetContactBuffer(max_time=0.2)
        self.l_foot_airtime = 0
        self.r_foot_airtime = 0
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        if self.randomize_dynamics or self.randomize_sensors:
            self.randomize_env()
            
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # Add noise to orientation and velocity if enabled
        qpos[3:7] += np.random.normal(0, self.imu_noise_std, size=4)  # quaternion noise
        qvel[0:3] += np.random.normal(0, self.vel_noise_std, size=3)  # linear velocity noise
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        
        # reset history
        if self.short_history_size > 0:
            self.short_history = np.zeros((self.short_history_size, self.short_history.shape[1]), dtype=np.float32)
        if self.long_history_size > 0:
            self.long_history = np.zeros((self.long_history_size, self.long_history.shape[1]), dtype=np.float32)
        if self.short_history_size > 0 or self.long_history_size > 0:
            obs = np.concatenate([obs, self.short_history.flatten(), self.long_history.flatten()])
            
        self.prev_joint_pos = self.data.qpos[7:].copy()  # Store previous joint positions for next step
        self.prev_actions = self.data.ctrl.copy()  # Store previous actions for next step
        self.original_height = self.data.qpos[2]  # Store original height for reward computation
        self.l_foot_orientation_command = self._get_geom_orientation(self.r_feet_geom)
        self.r_foot_orientation_command = self._get_geom_orientation(self.l_feet_geom)
        self.feet_contact_buffer.clear()  # Clear the buffer for feet contact
        self.l_foot_airtime = 0
        self.r_foot_airtime = 0
        self.t_last_perturbation = 0
        next_perturbation_config = self.random_config["t_perturbation"] or [0.1, 3]
        self.t_next_perturbation = np.random.uniform(next_perturbation_config[0], next_perturbation_config[1])
        return obs
    
    def randomize_env(self):
        # --- Dynamics randomization ---
        if self.randomize_dynamics:
            # Ground friction
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            friction_config = self.random_config["friction"] or [0.5, 1.5]
            self.model.geom_friction[floor_id][:] = np.random.uniform(friction_config[0], friction_config[1], size=3)

            # Joint damping
            joint_damping_config = self.random_config["joint_damping"] or [0.5, 1,5]
            self.model.dof_damping[:] *= np.random.uniform(joint_damping_config[0], joint_damping_config[1], size=self.model.nv)

            # Link mass & inertia
            mass_config = self.random_config["mass"] or [0.5, 1.5]
            self.model.body_mass[:] *= np.random.uniform(mass_config[0], mass_config[1], size=self.model.nbody)
            inertia_config = self.random_config["inertia"] or [0.7, 1.3]
            self.model.body_inertia[:] *= np.random.uniform(inertia_config[0], inertia_config[1], size=self.model.body_inertia.shape)

        # --- Sensor noise (applied during observation retrieval) ---
        if self.randomize_sensors:
            self.imu_noise_std = self.random_config["imu_noise"] or 0.01  # rad/s
            self.vel_noise_std = self.random_config["vel_noise"] or 0.02  # m/s
        else:
            self.imu_noise_std = 0.0
            self.vel_noise_std = 0.0

    def apply_random_perturbation(self):
        if not self.randomize_perturbations:
            return
        sim_t = self.data.time
        if sim_t - self.t_last_perturbation > self.t_next_perturbation:
            # Reset perturbation time
            self.t_last_perturbation = sim_t
            next_perturbation_config = self.random_config["t_perturbation"] or [0.1, 3]
            self.t_next_perturbation = np.random.uniform(next_perturbation_config[0], next_perturbation_config[1])
            
            # Apply random perturbation
            force_config = self.random_config["force"] or [-3, 3]
            force = np.random.uniform([force_config[0], force_config[0], force_config[0]], [force_config[1], force_config[1], force_config[1]])
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            # Get the 3D position of the torso body
            point = self.data.xpos[body_id]  # shape (3,)

            # Ensure all arrays are float64 and correct shape
            force = np.asarray(force, dtype=np.float64).reshape(3)
            torque = np.zeros(3, dtype=np.float64)
            qfrc_target = self.data.qfrc_applied  # already correct shape and dtype

            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                torque,
                point,
                body_id,
                qfrc_target
            )
            
    def step(self, action, render_ref_point=False):
        """
        Apply action and step the simulation.
        returns observation, reward, done, truncated, and info.
        observation is composed of a concatenation of positions and velocities.
        """
        # First, store the previous observation and actions in the history
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.short_history_size > 0:
            self.short_history = np.roll(self.short_history, -1, axis=0)
            self.short_history[-1] = np.concatenate([self.data.qpos, self.data.qvel, action]).astype(np.float32)
        if self.long_history_size > 0:
            self.long_history = np.roll(self.long_history, -1, axis=0)
            self.long_history[-1] = np.concatenate([self.data.qpos, self.data.qvel, action]).astype(np.float32)
        
        # Apply random perturbation
        self.apply_random_perturbation()
        
        # Step the simulation
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
            
        # Get observation
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # Add noise to orientation and velocity if enabled
        qpos[3:7] += np.random.normal(0, self.imu_noise_std, size=4)  # quaternion noise
        qvel[0:3] += np.random.normal(0, self.vel_noise_std, size=3)  # linear velocity noise
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        # Add history to the observation
        if self.long_history_size > 0 or self.short_history_size > 0:
            obs = np.concatenate([obs, self.short_history.flatten(), self.long_history.flatten()])
    
        terminated = self._is_terminated()
        reward = self._compute_reward()
        """if terminated:
            reward = -100  # Penalize falling"""
        
        return obs, reward, terminated, {}

    def _quaternion_distance(self, q1, q2, axis="yaw_pitch_roll"):
        # Take into account the axis of rotation wanted, yaw, pitch, roll or any combination of them
        # Remove rotations that are not in the axis of rotation wanted
        q1_euler = q1
        q2_euler = q2

        if len(q1) == 4:
            q1_euler = scipy.spatial.transform.Rotation.from_quat(q1).as_euler('zyx')
            q1_euler[2] += np.pi  # Add pi to the roll angle to account for the initial orientation of the robot
        if len(q2) == 4:
            q2_euler = scipy.spatial.transform.Rotation.from_quat(q2).as_euler('zyx')
            q2_euler[2] += np.pi  # Add pi to the roll angle to account for the initial orientation of the robot

        yaw_q1 = q1_euler[2] if "yaw" in axis else 0
        yaw_q2 = q2_euler[2] if "yaw" in axis else 0
        pitch_q1 = q1_euler[1] if "pitch" in axis else 0
        pitch_q2 = q2_euler[1] if "pitch" in axis else 0
        roll_q1 = q1_euler[0] if "roll" in axis else 0
        roll_q2 = q2_euler[0] if "roll" in axis else 0
        q1 = np.array([roll_q1, pitch_q1, yaw_q1])
        q2 = np.array([roll_q2, pitch_q2, yaw_q2])
        
        # Back to quaternions
        q1 = scipy.spatial.transform.Rotation.from_euler('zyx', q1)
        q2 = scipy.spatial.transform.Rotation.from_euler('zyx', q2)
        angle = q1.inv() * q2
        angle = angle.magnitude()  # Angle in radians
        # Normalize to [0, 1]
        distance = angle / np.pi  # Normalize to [0, 1]
        return distance  # Distance in radians
    
    def _feet_in_contact(self, foot):
        # Check if the foot is in contact with the ground
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == foot or contact.geom2 == foot:
                return True
        return False
    
    def _one_feet_in_contact(self):
        self.l_foot_contact = self._feet_in_contact(self.l_feet_geom)
        self.r_foot_contact = self._feet_in_contact(self.r_feet_geom)
        
        if self.l_foot_contact and not self.r_foot_contact:
            return True
        elif not self.l_foot_contact and self.r_foot_contact:
            return True
        else:
            return False
    
    def _get_geom_orientation(self, geom):
        # Get the orientation of a specific geometry
        # Extract the rotation matrix and convert it to a quaternion
        # Geom is the geometry id
        quaternion = self.data.xquat[geom]
        return quaternion
    
    def _get_geom_position(self, geom):
        # Get the position of a specific geometry
        # Geom is the geometry id
        geom_pos = self.data.geom_xpos[geom]
        return geom_pos
    
    def _compute_reward(self):
        
        # Reward function consists of:
        #Lets try setting the target speed at 0.6 m/s
        velocity_command = np.array([0.6, 0, 0])  # Desired velocity
        min_velocity = 0.1  # Minimum velocity to consider the robot moving
        height_command = 0.23  # Desired height of the robot
        vel_diff = np.linalg.norm(self.data.qvel[0:2] - velocity_command[0:2])
        vel = np.exp(-5*np.square(vel_diff))  # Penalize deviation from desired velocity
        if np.linalg.norm(self.data.qvel[0:2]) < min_velocity:
            vel = -1  # Penalize if the robot is not moving
        height = np.exp(-20*np.square(self.data.qpos[2] - height_command))
        
        # Action difference
        action = np.clip(self.data.ctrl, self.action_space.low, self.action_space.high)
        action_diff = np.sum(np.abs(action - self.prev_actions))  # Control effort
        action_diff = np.exp(-0.02*action_diff)  # Penalize action difference

        # Recuce torque
        max_torques = self.model.actuator_forcerange[:, 1]
        torques = self.data.actuator_force
        torque = np.exp(-0.02*np.sum(np.abs(torques)/max_torques)/len(torques))  # Penalize torque
        
        self.prev_joint_pos = self.data.qpos[7:].copy()  # Store previous joint positions for next step
        self.prev_actions = self.data.ctrl.copy()  # Store previous actions for next step
        
        base_accel = np.exp(-0.01*np.sum(np.abs(self.data.qacc[0:3])))  # Penalize acceleration of the base
        
        orientation_command = np.array([0, 0, 0])  # Desired orientation in yaw, pitch, roll
        # Term to keep in desired orientation
        yaw_orient = np.exp(-30*(self._quaternion_distance(self.data.qpos[3:7], orientation_command, axis="yaw")))
        # Term to keep straight
        pitch_roll_orient = np.exp(-30*(self._quaternion_distance(self.data.qpos[3:7], orientation_command, axis="pitch_roll")))
        
        # Feet orientation
        r_foot_orient = self._get_geom_orientation(self.r_feet_geom)
        l_foot_orient = self._get_geom_orientation(self.l_feet_geom)

        l_foot_orient_diff = np.abs(self._quaternion_distance(l_foot_orient, self.l_foot_orientation_command, axis="yaw_pitch_roll"))
        r_foot_orient_diff = np.abs(self._quaternion_distance(r_foot_orient, self.r_foot_orientation_command, axis="yaw_pitch_roll"))
        feet_orient = np.exp(-30*(l_foot_orient_diff + r_foot_orient_diff))
        
        # Keep COM between feet for stability
        com_pos = self.data.qpos[0:3]  # torso or base COM (x, y, z)
        l_foot_pos = self.data.geom_xpos[self.l_feet_geom][:3]
        r_foot_pos = self.data.geom_xpos[self.r_feet_geom][:3]
        
        feet_midpoint = 0.5 * (l_foot_pos + r_foot_pos)
        horizontal_offset = np.linalg.norm(com_pos[:2] - feet_midpoint[:2])
        torso_centering = np.exp(-20 * horizontal_offset**2)



        
        # The robot fell
        terminated = 0
        if self.data.qpos[2] < 0.2:
            terminated = 1
            
        # Multipliers for each term
        """step_reward = 0.1
        v_reward = 0.15
        height_reward = 0.05
        torque_reward = 0.02
        action_diff_reward = 0.02
        acceleration_reward = 0.1
        yaw_reward = 0.1
        pitch_roll_reward = 0.2
        terminated_reward = 0"""
        step_reward = 0.001
        v_reward = 0.15
        #height_reward = 0.05 / 2
        height_reward = 0
        torque_reward = 0.02 / 2
        action_diff_reward = 0.02 / 2
        acceleration_reward = 0.1 / 2
        yaw_reward = 0.02
        pitch_roll_reward = 0.04
        feet_orient_reward = 0.2
        torso_centering_reward = 0.1
        terminated_reward = -0.1 / 2
        
        # Compute reward
        forward_reward = \
            (vel * v_reward) + \
            step_reward + \
            (height * height_reward) + \
            (torque_reward * torque) + \
            (action_diff * action_diff_reward) + \
            (base_accel * acceleration_reward) + \
            (yaw_orient * yaw_reward) + \
            (pitch_roll_orient * pitch_roll_reward) + \
            (feet_orient * feet_orient_reward) + \
            (torso_centering * torso_centering_reward) + \
            (terminated * terminated_reward)
            
        return forward_reward

    def _is_terminated(self):
        # Terminate if the robot falls
        return self.data.qpos[2] < 0.2 # Z position too low
    
    def _initialize_renderer(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        self.window = glfw.create_window(self.viewport_width, self.viewport_height, "BasicEnv", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create a GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Initialize rendering context
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # Install GLFW mouse and keyboard callbacks
        glfw.set_cursor_pos_callback(self.window, self.free_camera.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.free_camera.mouse_button)
        glfw.set_scroll_callback(self.window, self.free_camera.scroll)
        
    def render(self):
        if not self.window or not self.context:
            raise RuntimeError("Render mode is not initialized. Set render_mode to 'human' or 'rgb_array'.")

        # Update scene and render
        viewport = mujoco.MjrRect(0, 0, self.viewport_width, self.viewport_height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)

        # Swap buffers and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        if glfw.window_should_close(self.window):
            self.close()
            return

    def close(self):
        if self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None