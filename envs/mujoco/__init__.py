from gymnasium.envs.registration import register

register(
    id="Mujoco-v0",
    entry_point="envs.mujoco_env:MujocoEnv",
    max_episode_steps=1000000,
)
