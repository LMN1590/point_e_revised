from typing import Any, Dict,Union

import torch
import torch.nn as nn

from .sdf import CrossAttentionPointCloudSDFModel
from .transformer import (
    CLIPImageGridPointDiffusionTransformer,
    CLIPImageGridUpsamplePointDiffusionTransformer,
    CLIPImagePointDiffusionTransformer,
    PointDiffusionTransformer,
    UpsamplePointDiffusionTransformer,
)
from ..config.model_dataclass import BaseModelConfig,ModelUpsampleConfig,SDFConfig


def model_from_config(config: Union[BaseModelConfig,ModelUpsampleConfig,SDFConfig], device: torch.device) -> nn.Module:
    config = config.copy()
    name = config.pop("name")
    if name == "PointDiffusionTransformer":
        return PointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImagePointDiffusionTransformer":
        return CLIPImagePointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImageGridPointDiffusionTransformer":
        return CLIPImageGridPointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "UpsamplePointDiffusionTransformer":
        return UpsamplePointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImageGridUpsamplePointDiffusionTransformer":
        return CLIPImageGridUpsamplePointDiffusionTransformer(
            device=device, dtype=torch.float32, **config
        )
    elif name == "CrossAttentionPointCloudSDFModel":
        return CrossAttentionPointCloudSDFModel(device=device, dtype=torch.float32, **config)
    raise ValueError(f"unknown model name: {name}")
