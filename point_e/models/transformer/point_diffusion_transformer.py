import torch
import torch.nn as nn

import math
from typing import List, Tuple

from ..blocks.mlp import MLP
from ..blocks.base_transformer import Transformer
from ..blocks.timestep_embedding import timestep_embedding

class PointDiffusionTransformer(nn.Module):
    def __init__(
        self, *,
        device:torch.device, dtype:torch.dtype,
        input_channels: int = 3, output_channels: int = 3,
        
        heads:int = 8, n_ctx:int = 1024,
        width:int = 512, init_scale:float = 0.25,
        layers:int = 12,
        time_token_cond:bool = False
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()
            
    def forward(self, x:torch.Tensor, t:torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])
    
    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token: h = h + emb[:, None] # emb [N,Width]
        
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ] # [N, 1, Width]
        if len(extra_tokens): h = torch.cat(extra_tokens + [h], dim=1) # [N, E+C , W]

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :] # remove extra tokens
            
        h = self.output_proj(h)
        return h.permute(0, 2, 1)
        