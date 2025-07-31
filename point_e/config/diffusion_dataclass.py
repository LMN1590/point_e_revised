from typing_extensions import TypedDict
from typing import List, Literal, Dict,Optional, Any

from yaml import safe_load

class BaseDiffusionConfig(TypedDict):
    channel_biases: List[float]
    channel_scales: List[float]
    mean_type: Literal['epsilon','x_start']
    schedule: Literal['cosine','linear']
    timesteps: int
    respacing: Optional[Any]
    
with open('point_e/config/diffusion_cfg.yaml', 'r') as file:
    diffusion_yaml_config = safe_load(file)
    
DIFFUSION_CONFIGS: Dict[str, BaseDiffusionConfig] = diffusion_yaml_config['DIFFUSION_CONFIGS']

if __name__ == "__main__":
    print(DIFFUSION_CONFIGS)