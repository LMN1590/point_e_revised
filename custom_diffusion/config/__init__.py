from .diffusion_dataclass import DIFFUSION_CONFIGS
from .model_dataclass import MODEL_CONFIGS

from typing_extensions import TypedDict
from typing import List,Literal

class PointEConfig(TypedDict):
    base_model_name: str
    device: str
    guidance_scale: List[float]
    num_points: List[int]
    sampling_mode: Literal['ddpm','ddim']