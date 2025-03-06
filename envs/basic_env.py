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

    def __init__(self, render_mode=None):
        # Path to robot XML
        xml_path = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.max_episode_steps = 1000000  # Large number of steps
        self.l_feet_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot")
        self.r_feet_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot")
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
        self.prev_actions = np.zeros(12)
        self.original_height = 0
        self.feet_contact_buffer = FeetContactBuffer(max_time=0.2)
        self.l_foot_airtime = 0
        self.r_foot_airtime = 0
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)  # Convert to float32
        self.prev_joint_pos = self.data.qpos[7:].copy()  # Store previous joint positions for next step
        self.prev_actions = self.data.ctrl.copy()  # Store previous actions for next step
        self.original_height = self.data.qpos[2]  # Store original height for reward computation
        self.feet_contact_buffer.clear()  # Clear the buffer for feet contact
        self.l_foot_airtime = 0
        self.r_foot_airtime = 0
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
        geom_mat = self.data.geom_xmat[geom]
        # Convert the rotation matrix to a quaternion
        rotation = scipy.spatial.transform.Rotation.from_matrix(geom_mat.reshape(3, 3))
        quaternion = rotation.as_quat()
        return quaternion
    
    def _get_geom_position(self, geom):
        # Get the position of a specific geometry
        # Geom is the geometry id
        geom_pos = self.data.geom_xpos[geom]
        return geom_pos
    
    def _compute_reward(self):
        """# Target commands
        standing = False
        velocity_command = np.array([0.6, 0, 0])  # Desired velocity
        if standing:
            velocity_command = np.array([0, 0, 0])
        orintation_command = np.array([0, 0, 0])  # Desired orientation in yaw, pitch, roll
        height_command = self.original_height  # Desired height of the robot

        l_foot_orientation_command = np.array([0, 0, 0]) # Desired orientation of the left foot in yaw, pitch, roll
        r_foot_orientation_command = np.array([0, 0, 0]) # Desired orientation of the right foot in yaw, pitch, roll
        l_foot_pos_command = np.array([0, 0, 0])  # Desired position of the left foot
        r_foot_pos_command = np.array([0, 0, 0])  # Desired position of the right foot

        # Difference between current and desired velocity
        vel_diff = np.linalg.norm(self.data.qvel[0:3] - velocity_command)
        velocity = np.exp(-5*np.square(vel_diff)) if not standing else np.exp(-5*vel_diff)

        # Term to keep in desired orientation
        yaw_orient = np.exp(-300*(self._quaternion_distance(self.data.qpos[3:7], orintation_command, axis="yaw")))
        # Term to keep straight
        pitch_roll_orient = np.exp(-30*(self._quaternion_distance(self.data.qpos[3:7], orintation_command, axis="pitch_roll")))

        # Feet contact with ground
        # Store feet contact in buffer
        if self._one_feet_in_contact():
            sim_time = self.data.time
            self.feet_contact_buffer.add(sim_time)
            
        feet_contact = 0
        if standing:
            feet_contact = 1
        else:
            if self.feet_contact_buffer.get(self.data.time) > 0:
                feet_contact = 1

        # Height difference from the desired height
        height = np.exp(-20*np.abs(self.data.qpos[2] - height_command))  # Height difference of the robot

        # Feet airtime
        l_feet_in_contact = self._feet_in_contact(self.l_feet_geom)
        r_feet_in_contact = self._feet_in_contact(self.r_feet_geom)
        #If the foot is in contact and the airtime is reseted, then this timestep is a touchdown
        l_foot_touchdown = l_feet_in_contact and self.l_foot_airtime == 0
        r_foot_touchdown = r_feet_in_contact and self.r_foot_airtime == 0
        feet_airtime = 0
        if standing:
            feet_airtime = 1
        else:
            if self.feet_contact_buffer.get(self.data.time) > 0:
                l = (self.data.time - self.l_foot_airtime -0.4) * l_foot_touchdown
                r = (self.data.time - self.r_foot_airtime -0.4) * r_foot_touchdown
                feet_airtime = l + r
        # When the foot is lifted, the airtime is reseted
        if not l_feet_in_contact and self.l_foot_airtime == 0:
            self.l_foot_airtime = self.data.time
        if not r_feet_in_contact and self.r_foot_airtime == 0:
            self.r_foot_airtime = self.data.time

        # Feet orientation
        l_foot_orientation_euler = scipy.spatial.transform.Rotation.from_quat(self._get_geom_orientation(self.l_feet_geom)).as_euler('zyx')
        r_foot_orientation_euler = scipy.spatial.transform.Rotation.from_quat(self._get_geom_orientation(self.r_feet_geom)).as_euler('zyx')
        # If yaw orientation commanded > 0:
        if np.abs(orintation_command[0]) > 0:
            # Only take into account roll and pitch
            l_foot_orientation_euler[0] = 0
            r_foot_orientation_euler[0] = 0
        # Calculate the difference
        l_foot_diff = np.sum(np.abs(l_foot_orientation_euler - l_foot_orientation_command))
        r_foot_diff = np.sum(np.abs(r_foot_orientation_euler - r_foot_orientation_command))
        feet_orientation = np.exp(-(l_foot_diff + r_foot_diff))

        # Foot position
        l_foot_pos = self._get_geom_position(self.l_feet_geom)
        r_foot_pos = self._get_geom_position(self.r_feet_geom)
        feet_position = 1
        if standing:
            l_foot_diff = np.linalg.norm(l_foot_pos - l_foot_pos_command)
            r_foot_diff = np.linalg.norm(r_foot_pos - r_foot_pos_command)
            feet_position = np.exp(-3*(l_foot_diff + r_foot_diff))
            
        # Base acceleration
        base_accel = np.sum(np.abs(self.data.qacc[0:3]))
        base_accel = np.exp(-0.01*base_accel)  # Penalize acceleration of the base

        # Action difference
        action = np.clip(self.data.ctrl, self.action_space.low, self.action_space.high)
        action_diff = np.sum(np.abs(action - self.prev_actions))  # Control effort
        action_diff = np.exp(-0.02*action_diff)  # Penalize action difference

        # Recuce torque
        max_torques = self.model.actuator_forcerange[:, 1]
        torques = self.data.actuator_force
        torque = np.exp(-0.01*np.sum(np.abs(torques)/max_torques)/len(torques))  # Penalize torque

        # Multipliers for each term
        vel_reward = 0.15
        yaw_reward = 0.1
        pitch_roll_reward = 0.2
        feet_contact_reward = 0.1
        height_reward = 0.05
        airtime_reward = 1 # Higher because it is sparse
        feet_orientation_reward = 0.05
        feet_position_reward = 0.05
        accel_reward = 0.1
        action_diff_reward = 0.02
        torque_reward = 0.02

        # For testing purposes, set rewards to 0
        feet_contact = 0
        feet_airtime = 0
        feet_orientation = 0
        feet_position = 0
        torque = 0
        accel_reward = 0
        action_diff_reward = 0
        # Compute reward
        forward_reward = \
            vel_reward * velocity + \
            yaw_reward * yaw_orient + \
            pitch_roll_reward * pitch_roll_orient + \
            feet_contact_reward * feet_contact + \
            height_reward * height + \
            airtime_reward * feet_airtime + \
            feet_orientation_reward * feet_orientation + \
            feet_position_reward * feet_position + \
            accel_reward * base_accel + \
            action_diff_reward * action_diff + \
            torque_reward * torque
        # print("Reward components:", velocity, yaw_orient, pitch_roll_orient, feet_contact, height, feet_airtime, feet_orientation, feet_position, base_accel, action_diff, torque)
        # print("Total reward:", forward_reward)
        return forward_reward
        """
        # Reward function consists of:
        # velx + fixed reward for each step - (height-desired height)^2
        # - (minimize control effort) - y^2 (deviation from y axis, keep straight)
        
        #Lets try setting the target speed at 0.6 m/s
        velocity_command = np.array([0.5, 0, 0])  # Desired velocity
        velx = self.data.qvel[0]  # Velocity in X direction
        velx = np.square(velocity_command[0]) - np.square(velx - velocity_command[0])
        height = np.square(self.original_height - self.data.qpos[2])  # Height difference of the robot
        servo_diff = np.sum(np.square(self.data.qpos[7:] - self.prev_joint_pos))  # Control effort
        self.prev_joint_pos = self.data.qpos[7:].copy()  # Store previous joint positions for next step
        axis_deviation = np.square(self.data.qpos[1])
        base_accel = np.sqrt(np.sum(self.data.qacc[1:3]**2))  # Acceleration of the base, penalize lateral acceleration
        
        orintation_command = np.array([0, 0, 0])  # Desired orientation in yaw, pitch, roll
        # Term to keep in desired orientation
        yaw_orient = np.exp(-300*(self._quaternion_distance(self.data.qpos[3:7], orintation_command, axis="yaw")))
        # Term to keep straight
        pitch_roll_orient = np.exp(-30*(self._quaternion_distance(self.data.qpos[3:7], orintation_command, axis="pitch_roll")))
        
        # Feet orientation
        l_foot_orientation_command = np.array([0, 0, 0]) # Desired orientation of the left foot in yaw, pitch, roll
        r_foot_orientation_command = np.array([0, 0, 0]) # Desired orientation of the right foot in yaw, pitch, roll
        r_foot_jaw = self._get_geom_orientation(self.r_feet_geom)
        l_foot_jaw = self._get_geom_orientation(self.l_feet_geom)
        l_foot_yaw_orient = np.exp(-300*(self._quaternion_distance(l_foot_jaw, l_foot_orientation_command, axis="yaw")))
        r_foot_yaw_orient = np.exp(-300*(self._quaternion_distance(r_foot_jaw, r_foot_orientation_command, axis="yaw")))
        feet_yaw_orient = l_foot_yaw_orient + r_foot_yaw_orient
        
        # The robot fell
        terminated = 0
        if self.data.qpos[2] < 0.2:
            terminated = 1
            
        # Multipliers for each term
        step_reward = 0.1
        vx_reward = 2
        height_reward = 0
        effort_reward = -0.02
        axis_reward = -3
        yaw_reward = -0.05
        pitch_roll_reward = -0.05
        acceleration_reward = -0.01
        terminated_reward = 0
        feet_yaw_orient_reward = -0.05
        
        # Compute reward
        forward_reward = \
            (velx * vx_reward) + \
            step_reward + \
            (height * height_reward) + \
            (servo_diff * effort_reward) + \
            (axis_deviation * axis_reward) + \
            (yaw_orient * yaw_reward) + \
            (pitch_roll_orient * pitch_roll_reward) + \
            (base_accel * acceleration_reward) + \
            (feet_yaw_orient * feet_yaw_orient_reward) + \
            (terminated * terminated_reward)
        #forward_reward = velx + step_reward
            
        return forward_reward

    def _is_terminated(self):
        # Terminate if the robot falls
        return self.data.qpos[2] < 0.2 or self.data.time > 19 # Z position too low
    
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