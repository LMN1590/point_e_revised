
from typing_extensions import TypedDict
from typing import Literal, Dict

class ModelConfig(TypedDict):
    sphere_radius: float
    num_points:int
    grid_res: int # poisson grid resolution
    psr_sigma: int
    apply_sigmoid: bool
    normal_normalize: bool
    
class TrainScheduleConfig(TypedDict):
    initial: float
    interval: int
    factor: float
    final: float
class TrainConfig(TypedDict):
    schedule:TrainScheduleConfig
    total_epochs:int
    
    w_chamfer:float
    w_chamfer_v_mesh: float
    w_chamfer_x_0: float
    w_inside: float
    w_outside: float
    
    n_sup_point: int
    n_grow_points: int
    subsample_vertex: bool
    l_weight: Dict

    exp_mesh:bool
    exp_pcl:bool
    dir_pcl:str
    dir_mesh:str
    
    resample_every: int

class DataConfig(TypedDict):
    data_type:Literal['point']
    n_views_per_iter: int
    zero_level:int
    
class SampleConfig(TypedDict):
    num_points:int
    density:float 
    voxel_size:float

class SAPConfig(TypedDict):
    model:ModelConfig
    train:TrainConfig
    data:DataConfig
    sample:SampleConfig
    
    device:str
    gradient_alpha:float