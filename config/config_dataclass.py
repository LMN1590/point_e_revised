from typing import TypedDict

from sap.config_dataclass import SAPConfig
from softzoo.configs.config_dataclass import FullConfig
from point_e.config import PointEConfig

class GeneralConfig(TypedDict):
    exp_name: str
    seed: int
    out_dir: str
    tensorboard_log_dir: str
    save_every_iter: int
    total_steps:int
        
    grad_scale: float
    grad_clamp: float
    calc_gradient: bool
    
    softzoo_config: FullConfig
    sap_config: SAPConfig
    pointe_config: PointEConfig