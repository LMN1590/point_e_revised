from typing import TYPE_CHECKING,Union

from ...configs.config_dataclass import ObjectiveForwardConfig,ObjectiveTrajectoryConfig

if TYPE_CHECKING:
    from ..base_env import BaseEnv
from .base import Base


class Dummy(Base):
    def __init__(self, env:'BaseEnv', config:Union[ObjectiveForwardConfig,ObjectiveTrajectoryConfig]):
        super().__init__(env, config)

    def reset(self):
        pass

    def get_obs(self, s):
        return None

    def get_reward(self, s):
        return 0

    @property
    def obs_shape(self):
        return None
