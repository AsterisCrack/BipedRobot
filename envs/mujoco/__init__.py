from gymnasium.envs.registration import register

register(
    id="Mujoco-v0",
    entry_point="envs.mujoco_env:MujocoEnv",
    max_episode_steps=1000000,
)

register(
    id="MujocoV2-v0",
    entry_point="envs.mujoco.biped_env_v2:BipedEnvV2",
    max_episode_steps=1000,
)
