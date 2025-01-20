import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import mujoco
from mujoco.glfw import glfw
from utils import free_camera_movement

class BasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        # Path to robot XML
        xml_path = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0  # Initialize positions
        self.data.qvel[:] = 0  # Initialize velocities
        mujoco.mj_forward(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)  # Convert to float32
        return obs, {}

    def step(self, action):
        # print("Action received:", action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)  # Convert to float32
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False  # Update if you implement truncation logic
        info = {}

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        # Example: Reward for moving forward
        forward_reward = self.data.qvel[0]  # Velocity in X direction
        return forward_reward

    def _is_terminated(self):
        # Example: Terminate if the robot falls
        return self.data.qpos[2] < -2  # Z position too low
    
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