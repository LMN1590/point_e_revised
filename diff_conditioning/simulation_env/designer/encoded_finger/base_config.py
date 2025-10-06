from typing import Tuple
from typing_extensions import TypedDict

class BaseConfig(TypedDict):
    complete_segment_path: str
    
    cylinder_color: Tuple[float,float,float]
    base_length: float
    radius: float
    
    finger_encoded_dim: int
    
    spline_range:Tuple[float,float]
    lengthen_range:Tuple[float,float]