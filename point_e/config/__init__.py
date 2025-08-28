from .diffusion_dataclass import DIFFUSION_CONFIGS
from .model_dataclass import MODEL_CONFIGS

from typing_extensions import TypedDict
from typing import List

class PointEConfig(TypedDict):
    base_model_name: str
    device: str
    guidance_scale: List[float]
    num_points: List[int]