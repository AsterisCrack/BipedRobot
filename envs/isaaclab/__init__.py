import gymnasium as gym

# Avoid importing Isaac Lab on module import so MuJoCo-only tools can still run.
try:
    import isaaclab  # noqa: F401
    import pxr  # noqa: F401
    _HAS_ISAACLAB = True
except ModuleNotFoundError:
    _HAS_ISAACLAB = False

if _HAS_ISAACLAB:
    from .biped_env_cfg import BipedEnvCfg as BipedRobotEnvCfg
    from .biped_env_cfg import BipedRobotV2EnvCfg
    from . import agents

    # Register Gym environments.
    gym.register(
        id="Isaac-BipedRobot-Direct-v0",
        entry_point="envs.isaaclab.biped_env:BipedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": BipedRobotEnvCfg,
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        },
    )

    gym.register(
        id="Isaac-BipedRobotV2-Direct-v0",
        entry_point="envs.isaaclab.biped_env:BipedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": BipedRobotV2EnvCfg,
        },
    )
