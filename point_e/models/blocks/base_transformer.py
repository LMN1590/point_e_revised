import torch
import torch.nn as nn

import math

from .attention import ResidualAttentionBlock

class Transformer(nn.Module):
    def __init__(
        self, *,
        device:torch.device, dtype:torch.dtype,
        heads:int, n_ctx:int,
        width:int, init_scale:float = 0.25,
        layers:int
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )
        
    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x