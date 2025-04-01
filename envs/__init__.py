from gymnasium.envs.registration import register

register(
    id="Basic-v0",
    entry_point="envs.basic_env:BasicEnv",
    max_episode_steps=1000000,
)

register(
    id="Advanced-v0",
    entry_point="envs.advanced_env:AdvancedEnv",
    max_episode_steps=1000000,
)
