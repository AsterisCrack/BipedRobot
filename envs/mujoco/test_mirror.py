import mujoco
import mujoco.viewer
import numpy as np
import time
from envs.utils.mirroring import mirror_observation

MODEL_PATH = "envs/assets/robot/Robot_description/urdf/robot_mujoco.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Get joint count
n_joints = model.nu  # exclude free joint



# Viewer setup
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Initial timestamp
    last_mirror_time = time.time()

    # Run simulation
    while viewer.is_running():
        now = time.time()

        # Randomize qpos at start
        data.qpos[0:13] = np.random.uniform(low=-0.5, high=0.5, size=13)
        data.qpos[13:] = np.random.uniform(low=0, high=0.5, size=n_joints//2)

        # Reset qvel to zero
        data.qvel[:] = 0

        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0)  # Show initial posture for 1s

        # Then loop: mirror every 1s
        while viewer.is_running():
            if time.time() - last_mirror_time > 1:
                # Get current state as observation
                obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
                mirrored_obs = mirror_observation(obs, n_joints)

                # Update state
                data.qpos[:] = mirrored_obs[:model.nq]
                data.qvel[:] = mirrored_obs[model.nq:]
                mujoco.mj_forward(model, data)

                last_mirror_time = time.time()

            viewer.sync()
            time.sleep(0.01)

