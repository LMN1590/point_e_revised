from typing import TYPE_CHECKING,Union

from ...configs.config_dataclass import ObjectiveForwardConfig,ObjectiveTrajectoryConfig,ThrowObjectConfig

if TYPE_CHECKING:
    from ..base_env import BaseEnv

class Base:
    def __init__(self, env:'BaseEnv', config:Union[ObjectiveTrajectoryConfig,ObjectiveForwardConfig,ThrowObjectConfig]):
        self.env = env
        self.config = config

    def reset(self):
        raise NotImplementedError

    def get_obs(self, s):
        raise NotImplementedError

    def get_reward(self, s):
        raise NotImplementedError

    def get_done(self):
        return False

    @property
    def obs_shape(self):
        raise NotImplementedError
