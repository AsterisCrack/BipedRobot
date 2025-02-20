import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import mujoco
from mujoco.glfw import glfw
from utils import free_camera_movement
import scipy.spatial.transform

class BasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        # Path to robot XML
        xml_path = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.max_episode_steps = 1000000  # Large number of steps
        self.name = "BasicEnv"

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
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)  # Convert to float32
        self.prev_joint_pos = self.data.qpos[7:].copy()  # Store previous joint positions for next step
        self.original_height = self.data.qpos[2]  # Store original height for reward computation
        return obs

    def step(self, action, render_ref_point=False):
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
            
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)  # Convert to float32
        reward = self._compute_reward()
        terminated = self._is_terminated()
        """if terminated:
            reward = -100  # Penalize falling"""
        
        return obs, reward, terminated, {}

    def _compute_reward(self):
        # Reward function consists of:
        # velx + fixed reward for each step - (height-desired height)^2
        # - (minimize control effort) - y^2 (deviation from y axis, keep straight)
        velx = self.data.qvel[0]  # Velocity in X direction
        height = np.square(self.original_height - self.data.qpos[2])  # Height difference of the robot
        servo_diff = np.sum(np.square(self.data.qpos[7:] - self.prev_joint_pos))  # Control effort
        self.prev_joint_pos = self.data.qpos[7:].copy()  # Store previous joint positions for next step
        axis_deviation = np.square(self.data.qpos[1])
        
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
        return self.data.qpos[2] < 0.2  # Z position too low
    
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