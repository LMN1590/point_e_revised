from typing_extensions import TypedDict
from typing import List, Literal, Dict,Optional, Any

from yaml import safe_load

from ..diffusion.gaussian_diffusion.const import DIFFUSION_MEAN_TYPE, DIFFUSION_VAR_TYPE, DIFFUSION_LOSS_TYPE

class BaseDiffusionConfig(TypedDict):
    channel_biases: List[float]
    channel_scales: List[float]
    mean_type: Literal['epsilon','x_start','x_prev']
    schedule: Literal['cosine','linear']
    timesteps: int
    respacing: Optional[Any]
    k: int                                      # Number of MCMC sampling in one denoising step
    condition_threshold: int                    # Threshold for t<threshold to start simulation conditioning
    
with open('custom_diffusion/config/diffusion_cfg.yaml', 'r') as file:
    diffusion_yaml_config = safe_load(file)
    
DIFFUSION_CONFIGS: Dict[str, BaseDiffusionConfig] = diffusion_yaml_config['DIFFUSION_CONFIGS']

if __name__ == "__main__":
    print(DIFFUSION_CONFIGS)