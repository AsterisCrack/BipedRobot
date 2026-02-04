import gymnasium as gym

from .biped_env_cfg import BipedEnvCfg
from .biped_env_cfg_v2 import BipedEnvCfg as BipedEnvCfgV2
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Biped-Direct-v0",
    entry_point="envs.isaaclab.biped_env:BipedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-BipedV2-Direct-v0",
    entry_point="envs.isaaclab.biped_env_v2:BipedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedEnvCfgV2,
    },
)
