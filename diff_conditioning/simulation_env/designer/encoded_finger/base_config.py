from typing import Tuple,List
from typing_extensions import TypedDict

class SegmentConfig(TypedDict):
    complete_segment_path: str
    
    cylinder_color: Tuple[float,float,float]
    base_length: float
    radius: float
    
    finger_encoded_dim: int
    
    spline_range:Tuple[float,float]
    lengthen_range:Tuple[float,float]
    rotation_range: List[Tuple[float,float,float]]
    softness_range:Tuple[float,float]
    actuation_strength_range:Tuple[float,float]
    suction_range:Tuple[float,float]
    
class BaseConfig(TypedDict):
    fixed_base_path:str
    finger_radius: float # Fingers are arranged around in a circle. This value determines the radius of that circle

class Config(TypedDict):
    fixed_base_config:BaseConfig
    segment_config: SegmentConfig
    