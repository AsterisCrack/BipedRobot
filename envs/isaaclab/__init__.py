import gymnasium as gym

from .biped_env import BipedEnv
from .biped_env_cfg import BipedEnvCfg
from .basic_biped_env import BasicBipedEnv
from .basic_biped_env_cfg import BasicBipedEnvCfg
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
    id="Isaac-BasicBiped-Direct-v0",
    entry_point="envs.isaaclab.basic_biped_env:BasicBipedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BasicBipedEnvCfg,
    },
)
