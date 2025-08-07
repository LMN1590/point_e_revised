import taichi as ti

from typing import Optional, Dict, Literal
import os

from . import ENV_CONFIGS_DIR
from .base_env import BaseEnv
from ..configs.config_dataclass import SimulatorConfig


@ti.data_oriented
class DummyEnv(BaseEnv):
    def __init__(
        self, 
        cfg_file: str,
        out_dir: str, 
        device:Optional[Literal['numpy','torch_cpu','torch_gpu']] = 'numpy',
        cfg_kwargs: Optional[Dict] = dict()
    ):
        cfg_file = os.path.join(ENV_CONFIGS_DIR, cfg_file)
        super().__init__(cfg_file, out_dir, device, cfg_kwargs)

    def get_reward(self):
        return 0.
