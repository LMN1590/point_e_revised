import torch
import torch.nn as nn

import math
from typing import List, Tuple, Optional, Dict, Any, Iterable

from ..blocks.mlp import MLP
from ..blocks.base_transformer import Transformer
from ..blocks.timestep_embedding import timestep_embedding,finger_segment_geo_embedding
from ..object_encoder import EncoderPlaceholder

class GripperRepDiffusionTransformer(nn.Module):
    # region Initialization
    def __init__(
        self,*,
        device:torch.device, dtype:torch.dtype,
        input_channels: int = 11, output_channels: int = 11,
        
        heads:int = 8, n_ctx:int = 40,
        width:int = 512, init_scale:float = 0.25,
        layers:int = 12,
        cond_drop_prob: float = 0.0,
        time_token_cond:bool = False,
        
        # portion for object encoding
    ):
        super().__init__()
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.cond_drop_prob = cond_drop_prob
        
        self.num_fingers = None
        self.max_num_segments = None
        self.gripper_dim = None
        self.total_dim = None
        self.gripper_pos_encoding = None
        
        self.object_encoder = self._init_object_encoder()
        
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.object_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=self.object_encoder.feature_dim, device=device, dtype=dtype
            ),
            nn.Linear(self.object_encoder.feature_dim, width, device=device, dtype=dtype),
        )
        
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond) + self.object_encoder.n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj_gripper = nn.Linear(width, output_channels-2,device=device,dtype=dtype)
        self.output_proj_mask = nn.Linear(width,2,device=device,dtype=dtype)
        
        with torch.no_grad():
            self.output_proj_gripper.weight.zero_()
            self.output_proj_gripper.bias.zero_()
            
            self.output_proj_mask.weight.zero_()
            self.output_proj_mask.bias.zero_()
    
    def _init_object_encoder(self):
        return EncoderPlaceholder()
    
    def _init_fingers_topo(
        self,
        gripper_dim: int = 10,
        max_num_segments: int = 10,
        num_fingers: int = 4
    ):
        self.finger_dim = gripper_dim
        self.total_dim = gripper_dim + 1
        self.max_num_segments = max_num_segments
        self.num_fingers = num_fingers
        
        assert self.n_ctx == self.num_fingers*self.max_num_segments
        assert self.input_channels == self.total_dim
        
        self.gripper_pos_encoding = finger_segment_geo_embedding(
            self.num_fingers,
            self.max_num_segments,
            self.total_dim
        ) # TC
        
    # endregion
            
    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # _ = batch_size
        with torch.no_grad():
            return dict(
                embeddings=self.object_encoder.encode(
                    model_kwargs["objects"], batch_size
                ).to(self.device) # [B,feature_dim,T]
            )
        
    def forward(
        self, x:torch.Tensor, t:torch.Tensor,
        embeddings:torch.Tensor,
        **model_kwargs
    ):
        """
        :param x: an [B x C x T] tensor.
        :param t: an [B] tensor.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [B x C' x T] tensor.
        """
        assert embeddings is not None, "must specify embeddings"
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        object_encoded = embeddings
        assert object_encoded.shape[0] == x.shape[0]
        
        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            object_encoded = object_encoded * mask[:, None, None].to(object_encoded)
        object_encoded = object_encoded.permute(0,2,1) # BCT -> BTC
        object_embed = self.object_embed(object_encoded)
        
        cond = [(t_embed, self.time_token_cond), (object_embed, True)]
        return self._forward_with_cond(x, cond)
    
    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        assert self.gripper_pos_encoding is not None
        h = x + self.gripper_pos_encoding[None,:,:].permute(0,2,1).to(x) # BCT
        
        h = self.input_proj(h.permute(0, 2, 1))  # BCT -> BTC
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
            
        gripper_h = self.output_proj_gripper(h)
        mask_h = self.output_proj_mask(h)
        return torch.cat([
            gripper_h[:,:,:(self.output_channels-2)//2],
            mask_h[:,:,:1],
            gripper_h[:,:,(self.output_channels-2)//2:],
            mask_h[:,:,1:],
        ],dim=-1).permute(0, 2, 1) # BCT