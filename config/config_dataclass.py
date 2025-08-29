from typing import TypedDict,List

from sap.config_dataclass import SAPConfig
from softzoo.configs.config_dataclass import FullConfig
from point_e.config import PointEConfig

class ConditioningConfig(TypedDict):
    name:str
    grad_scale:float
    grad_clamp:float
    calc_gradient:bool
    logging_bool: bool

class GeneralConfig(TypedDict):
    exp_name: str
    seed: int
    out_dir: str
    tensorboard_log_dir: str
    save_every_iter: int
    total_steps:int
    
    cond_config: List[ConditioningConfig]
    cond_overall_logging: bool
    
    embedding_path: str # Path to the precomputed embeddings
    
    softzoo_config: FullConfig
    sap_config: SAPConfig
    pointe_config: PointEConfig