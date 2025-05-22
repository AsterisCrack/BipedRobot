import yaml

class NoConfig:
    def __init__(self, config=None):
        pass

    def __getitem__(self, key):
        return NoConfig()

    def __bool__(self):
        return False
    
class ConfigSubdict:
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, key):
        if key not in self.data:
            return None
        if type(self.data[key]) == dict:
            return ConfigSubdict(self.data[key])
        return self.data[key]
    
class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self._convert_scientific_notation(self.config)

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def copy_in_file(self, config_path):
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
            
    def __getitem__(self, key):
        if key not in self.config:
            return None
        if type(self.config[key]) == dict:
            return ConfigSubdict(self.config[key])
        return self.config[key]
    
    def _convert_scientific_notation(self, obj):
        """
        Recursively explore the config and convert strings in scientific notation
        (e.g., '1e5', '-2e-3') into integers or floats.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._convert_scientific_notation(value)
        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = self._convert_scientific_notation(obj[i])
        elif isinstance(obj, str):
            try:
                return float(obj)
            except:
                pass
        return obj
    
# Functions to mirror observations and actions for data augmentation
import numpy as np

def mirror_observation(obs, n_joints):
    """
    Mirrors a MuJoCo observation over the Y-axis.
    
    Parameters:
    - obs: np.array, shape=(7 + n_joints + 6 + n_joints,)
    - n_joints: int, number of joints (must be even to mirror properly)
    
    Returns:
    - mirrored_obs: np.array, mirrored observation
    """
    """obs = obs.copy()
    
    # Indices
    pos_start = 0
    quat_start = 3
    joint_start = 7
    vel_start = 7 + n_joints
    joint_vel_start = vel_start + 6
    
    # 1. Mirror base position (negate y)
    obs[1] *= -1
    
    # 2. Mirror orientation quaternion over Y-axis (XZ plane)
    qx, qy, qz, qw = obs[quat_start:quat_start+4]
    obs[quat_start:quat_start+4] = [-qx, qy, -qz, qw]
    
    # 3. Swap joint angles (first half <-> second half)
    joint_angles = obs[joint_start:vel_start]
    half = n_joints // 2
    obs[joint_start:vel_start] = np.concatenate([joint_angles[half:], joint_angles[:half]])*-1
    
    # Invert joints on y-axis
    obs[9] *= -1  # r_hip_x
    obs[10] *= -1  # r_knee
    obs[11] *= -1  # r_ankle_y
    obs[15] *= -1  # l_hip_x
    obs[16] *= -1  # l_knee
    obs[17] *= -1  # l_ankle_y
    
    # 4. Mirror linear velocity (negate vy)
    obs[vel_start+1] *= -1  # vy
    
    # 5. Mirror angular velocity (negate wx, wz)
    obs[vel_start+3] *= -1  # wx
    obs[vel_start+5] *= -1  # wz
    
    # 6. Swap joint velocities (first half <-> second half)
    joint_vels = obs[joint_vel_start:]
    obs[joint_vel_start:] = np.concatenate([joint_vels[half:], joint_vels[:half]])*-1
    
    # Invert joint velocities on y-axis
    obs[joint_vel_start+2] *= -1  # r_hip_x
    obs[joint_vel_start+3] *= -1  # r_knee
    obs[joint_vel_start+4] *= -1  # r_ankle_y
    obs[joint_vel_start+8] *= -1  # l_hip_x
    obs[joint_vel_start+9] *= -1  # l_knee
    obs[joint_vel_start+10] *= -1  # l_ankle_y"""
    
    return obs

def mirror_action(action, n_joints):
    """
    Mirrors a MuJoCo action over the Y-axis.
    
    Parameters:
    - action: np.array, shape=(n_joints,)
    - n_joints: int, number of joints (must be even to mirror properly)
    
    Returns:
    - mirrored_action: np.array, mirrored action
    """
    """action = action.copy()
    
    # 1. Swap joint angles (first half <-> second half)
    half = n_joints // 2
    action = np.concatenate([action[half:], action[:half]])*-1
    # Mirror joint angles on y-axis
    action[0] *= -1  # r_hip_x
    action[1] *= -1  # r_knee
    action[2] *= -1  # r_ankle_y
    action[6] *= -1  # l_hip_x
    action[7] *= -1  # l_knee
    action[8] *= -1  # l_ankle_y"""
    
    return action
