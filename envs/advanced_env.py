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
        
class AdvancedEnv(gym.Env):
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
        self.name = "AdvancedEnv"
        self.original_height = self.data.qpos[2]
        self.steps_in_target = 0
        
        # Variables for the random target
        self.reached_target_threshold = 0.05
        self.target_body_id = self.model.body("visual_target").id
        self._reset_target()

        # Observation space: Full state (joint positions and velocities)
        obs_dim = self.model.nq + self.model.nv + 1 # Positions + velocities + Target position
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # Action space: Control position of servos
        action_dim = self.model.nu
        self.action_space = Box(low=-np.pi, high=np.pi, shape=(action_dim,), dtype=np.float32)

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
        self.feet_contact_buffer = FeetContactBuffer(max_time=0.2)
        self.l_foot_airtime = 0
        self.r_foot_airtime = 0
        self.reset()
    
    def _get_random_target(self):
        # Get random target vector of module 1
        # Generate a random angle
        angle = np.random.uniform(0, 2 * np.pi)
        
        return angle
    
    def _reset_target(self):
        self.target = self._get_random_target()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # Reset target position
        self._reset_target()
        angle_difference = np.array([np.abs(self.data.qpos[3] - self.target)])  # Yaw angle difference to the target
        obs = np.concatenate([self.data.qpos, self.data.qvel, angle_difference]).astype(np.float32)  # Convert to float32
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
        
        angle_difference = np.array([np.abs(self.data.qpos[3] - self.target)])  # Yaw angle difference to the target
        obs = np.concatenate([self.data.qpos, self.data.qvel, angle_difference]).astype(np.float32)
    
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
        # Reward function consists of:
        # velx + fixed reward for each step - (height-desired height)^2
        # - (minimize control effort) - y^2 (deviation from y axis, keep straight)
        
        # Set training stage:
        train_stage = 1
        
        #Lets try setting the target speed at 0.6 m/s
        height_command = 0.23  # Desired height of the robot

        # Set the orientation command to point towards the target
        orintation_command = np.array([
            np.arctan2(np.sin(self.target), np.cos(self.target)),  # Yaw angle to face the target
            0,  # Pitch angle (not used in this case)
            0   # Roll angle (not used in this case)
        ])
        
        # Position, this one is linear
        """pos_diff = np.linalg.norm(self.data.qpos[0:2] - position_command)
        position = self.max_target_distance - pos_diff/self.max_target_distance # Penalize deviation from desired position"""
        
        # Velocity
        # Velocity command is a vector pointing towards target with module 0.6
        desired_velocity = 0.6
        direction_vector = np.array([np.cos(self.target), np.sin(self.target), 0])  # Direction vector towards the target in x, y, z coordinates
        velocity_command = direction_vector / np.linalg.norm(direction_vector) * desired_velocity
        vel_diff = np.linalg.norm(self.data.qvel[0:2] - velocity_command[0:2])
        vel = np.exp(-5*np.square(vel_diff))  # Penalize deviation from desired velocity
        
        # Height
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
    
        # Term to keep in desired orientation
        yaw_difference = np.abs(self.data.qpos[3] - orintation_command[0])
        yaw_orient = np.exp(-30*(yaw_difference/np.pi))
        # Term to keep straight
        pitch_roll_orient = np.exp(-30*(self._quaternion_distance(self.data.qpos[3:7], orintation_command, axis="pitch_roll")))
        
        # Feet orientation
        """l_foot_orientation_command = np.array([0, 0, 0]) # Desired orientation of the left foot in yaw, pitch, roll
        r_foot_orientation_command = np.array([0, 0, 0]) # Desired orientation of the right foot in yaw, pitch, roll
        r_foot_jaw = self._get_geom_orientation(self.r_feet_geom)
        l_foot_jaw = self._get_geom_orientation(self.l_feet_geom)
        l_foot_yaw_orient = np.exp(-300*(self._quaternion_distance(l_foot_jaw, l_foot_orientation_command, axis="yaw")))
        r_foot_yaw_orient = np.exp(-300*(self._quaternion_distance(r_foot_jaw, r_foot_orientation_command, axis="yaw")))
        feet_yaw_orient = l_foot_yaw_orient + r_foot_yaw_orient"""
        
        looking_at_target = self._lookig_at_target(threshold=0.1)  # Check if the robot is looking at the target
        # The robot fell
        terminated = 0
        if self.data.qpos[2] < 0.2:
            terminated = 1
        
        # Check if reached the target:
        reached_target = self._reached_target(train_stage)
        # Multipliers for each term
        step_reward = 0.001
        position, position_reward = 0, 0
        v_reward = 0.15
        height_reward = 0.05
        torque_reward = 0.02
        action_diff_reward = 0.02
        acceleration_reward = 0.1
        yaw_reward = 0.02
        pitch_roll_reward = 0.02
        terminated_reward = -0.1
        target_reward = 1
        
        if train_stage == 1:
            # Term to keep in desired orientation
            yaw_difference = np.abs(self.data.qpos[3] - orintation_command[0])
            yaw_orient = np.exp(-5*(yaw_difference/np.pi))
            # In this stage we only want it to turn in place and learn to balance
            vel = -np.linalg.norm(self.data.qvel[0:2])  # Penalize forward velocity to keep it in place
            position = np.exp(-5*np.square(np.linalg.norm(self.data.qpos[0:2] - [0,0])))  # Penalize deviation from center
            # print("Yaw difference:", yaw_difference, "Looking at target:", looking_at_target, "Angle:", self.data.qpos[3], "Target:", self.target)

            # Multipliers for each term
            """step_reward = 0.001
            position_reward = 0.01
            v_reward = 0.15
            height_reward = 0.05
            torque_reward = 0
            action_diff_reward = 0
            acceleration_reward = 0.1
            yaw_reward = 0.1
            pitch_roll_reward = 0.02
            terminated_reward = -0.1
            target_reward = 1"""
            step_reward = 0
            position_reward = 0
            v_reward = 0
            height_reward = 0
            torque_reward = 0
            action_diff_reward = 0
            acceleration_reward = 0
            yaw_reward = 0.02
            pitch_roll_reward = 0
            terminated_reward = 0
            target_reward = 10
            looking_at_target_reward = 0.1
        
        
        # Compute reward
        forward_reward = \
            (vel * v_reward) + \
            step_reward + \
            (position * position_reward) + \
            (height * height_reward) + \
            (torque_reward * torque) + \
            (action_diff * action_diff_reward) + \
            (base_accel * acceleration_reward) + \
            (yaw_orient * yaw_reward) + \
            (pitch_roll_orient * pitch_roll_reward) + \
            (reached_target * target_reward) + \
            (looking_at_target * looking_at_target_reward) + \
            (terminated * terminated_reward)
            #(position * position_reward) + \
                
        #forward_reward = velx + step_reward
            
        return forward_reward

    def _lookig_at_target(self, threshold=0.1):
        # Check if the robot is looking at the target
        # Compute the yaw angle to face the target
        yaw_difference = np.abs(self.data.qpos[3] - self.target)
        # Check if the robot is facing the target within a certain threshold
        return yaw_difference < threshold
            
    def _reached_target(self, train_stage=2, target_steps=100, threshold=0.1):
        if train_stage == 1:
            # In stage 1, we only want to check if the robot is stably looking at the target
            if self._lookig_at_target(threshold=threshold):
                # If the robot is facing the target, we can consider it as having reached the target
                self.steps_in_target += 1
                # If the robot has been facing the target for a certain number of steps, consider it as reached
                if self.steps_in_target >= target_steps:
                    self.steps_in_target = 0
                    self._reset_target()
                    return True
                else:
                    # If the robot is facing the target but hasn't reached the target yet, return False
                    return False
            else:
                # If the robot is not facing the target, reset the steps in target counter
                self.steps_in_target = 0
                return False
        # In stage 2, we check if the robot has reached the target position
        else:
            # Check if the robot has reached the target
            robot_pos = self.data.qpos[0:2]
            target_pos = self.target
            distance = np.linalg.norm(robot_pos - target_pos)
            reached = distance < self.reached_target_threshold
            if reached:
                self._reset_target()
            return reached
    
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
        # Reset the position of the target object
        #self.data.xpos[self.target_body_id] = np.array([self.target[0], self.target[1], self.original_height])
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