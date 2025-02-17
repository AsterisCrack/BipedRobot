import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import mujoco
from mujoco.glfw import glfw
from utils import free_camera_movement
from scipy.spatial.transform import Rotation as R

class Data:
    def __init__(self, env, qpos=None, qvel=None):
        self.qpos = qpos if qpos is not None else np.zeros(env.model.nq)
        self.qvel = qvel if qvel is not None else np.zeros(env.model.nv)
        
class BasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        # Path to robot XML
        xml_path = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"
        self.free_joint_displacement = 0.3  # Displacement of the free joint in the torso
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # Observation space: Full state (joint positions and velocities)
        obs_dim = self.model.nq + self.model.nv  # Positions + velocities
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # Action space: Control position of servos
        action_dim = self.model.nu
        self.action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

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
        self.original_height = 0
        self.corrected_data = Data(self)
        self.reset()
    
    def _correct_free_joint(self, data_original):
        
        data = Data(self, self.data.qpos.copy(), self.data.qvel.copy())
        # Extract position and quaternion of the torso
        qpos = data.qpos  # Full state
        torso_pos = qpos[:3]  # Current free joint position
        torso_quat = qpos[3:7]  # Rotation quaternion
        # Define the displacement vector in local coordinates
        local_offset = np.array([0, 0, -self.free_joint_displacement])  # Move up by d in torso's local +Z axis
        # Ensure the quaternion is valid (normalize if necessary)
        norm = np.linalg.norm(torso_quat)
        if norm == 0:
            return
        torso_quat /= norm  # Normalize the quaternion
        # Convert local offset to global coordinates using the quaternion rotation
        global_offset = R.from_quat(torso_quat).apply(local_offset)

        # Apply the correction
        corrected_pos = torso_pos + global_offset

        # Update qpos
        data.qpos[:3] = corrected_pos  # Move the free joint up
        self.corrected_data = data  # Store corrected data
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0  # Initialize positions
        self.data.qvel[:] = 0  # Initialize velocities
        mujoco.mj_forward(self.model, self.data)
        self._correct_free_joint(self.data)  # Move the free joint up
        obs = np.concatenate([self.corrected_data.qpos, self.corrected_data.qvel]).astype(np.float32)  # Convert to float32
        self.prev_joint_pos = self.corrected_data.qpos[7:].copy()  # Store previous joint positions for next step
        self.original_height = self.corrected_data.qpos[2]  # Store original height for reward computation
        return obs, {}

    def step(self, action):
        """
        Apply action and step the simulation.
        returns observation, reward, done, truncated, and info.
        observation is composed of a concatenation of positions and velocities.
        Positions, length = 7 + n joints.
            First 3 are x, y, z, next 4 are rotation quaternion, finally n joints with the position of each joint.
        Velocities, length = 6 + n joints.
            First 3 are linear velocities, next 3 are angular velocities, finally n joints with the velocity of each joint.
        """
        # print("Action received:", action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._correct_free_joint(self.data)  # Move the free joint up
        
        obs = np.concatenate([self.corrected_data.qpos, self.corrected_data.qvel]).astype(np.float32)  # Convert to float32
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False  # Update if you implement truncation logic
        info = {}

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """Observation: [ 8.88922950e-04  8.71229102e-04 -1.61353627e-03  9.99990106e-01
        3.67753324e-03 -2.47385376e-03 -3.08103248e-04  1.07131324e-04
        -5.55411307e-03  1.77131838e-03  2.08842405e-03  2.39029061e-03
        -1.33043516e-03 -1.63827252e-04 -1.45398232e-03  1.44785584e-03
        1.74693763e-03  2.38776486e-03 -5.59406681e-03  1.13315750e-02
        -4.62242728e-03 -1.28024130e-03 -2.43785023e-03 -1.40542369e-02
        -9.40201432e-02 -8.93840261e-05  1.18226651e-02  2.17969771e-02
        5.71652614e-02  1.00348510e-01  5.91178350e-02  1.35074295e-02
        1.43485302e-02  2.09552273e-02  9.93347075e-03  4.23588557e-03
        -3.58647597e-03]"""
        # Reward function consists of:
        # velx + fixed reward for each step - (height-desired height)^2
        # - (minimize control effort) - y^2 (deviation from y axis, keep straight)
        velx = self.corrected_data.qvel[0]  # Velocity in X direction
        height = np.square(self.original_height - self.corrected_data.qpos[2])  # Height difference of the robot
        servo_diff = np.sum(np.square(self.corrected_data.qpos[7:] - self.prev_joint_pos))  # Control effort
        self.prev_joint_pos = self.corrected_data.qpos[7:].copy()  # Store previous joint positions for next step
        axis_deviation = np.square(self.corrected_data.qpos[1])
        
        # Multipliers for each term
        step_reward = 0.0625
        vx_reward = 1
        height_reward = -50
        effort_reward = -0.02
        axis_reward = -3
        
        # Compute reward
        forward_reward = \
            (velx * vx_reward) + \
            step_reward + \
            (height * height_reward) + \
            (servo_diff * effort_reward) + \
            (axis_deviation * axis_reward)
        #forward_reward = velx + step_reward
            
        return forward_reward

    def _is_terminated(self):
        # Example: Terminate if the robot falls
        return self.corrected_data.qpos[2] < 0.15  # Z position too low
    
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