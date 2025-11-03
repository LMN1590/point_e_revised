from typing_extensions import TypedDict
from typing import List, Literal, Dict, Union

from yaml import safe_load

class BaseModelConfig(TypedDict):
    cond_drop_prob: float
    heads: int
    init_scale: float
    input_channels: int
    layers: int
    n_ctx: int
    name: str
    output_channels: int
    time_token_cond: bool
    token_cond: bool
    width: int

class ModelUpsampleConfig(BaseModelConfig):
    channel_biases: List[float]
    channel_scales: List[float]
    cond_ctx: int

class SDFConfig(BaseModelConfig):
    decoder_heads: int
    decoder_layers: int
    encoder_heads: int
    encoder_layers: int
    
with open('point_e/config/model_cfg.yaml', 'r') as file:
    model_yaml_config = safe_load(file)
    
MODEL_CONFIGS: Dict[str, BaseModelConfig] = model_yaml_config['MODEL_CONFIGS']

if __name__ == "__main__":
    print(MODEL_CONFIGS)