import gymnasium as gym
import numpy as np
import mujoco
from mujoco.glfw import glfw
from utils import free_camera_movement
import scipy.spatial.transform

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

class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # Rendering attributes
        self.window = None
        self.context = None
        self.viewport_width = 1920
        self.viewport_height = 1080
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.free_camera = free_camera_movement.FreeCameraMovement(self.model, self.cam, self.scene)
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._initialize_renderer()

    def _initialize_renderer(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        self.window = glfw.create_window(self.viewport_width, self.viewport_height, "Env", None, None)
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
        
    def _is_terminated(self):
        # Terminate if the robot falls
        return self.data.qpos[2] < 0.2 # Z position too low
