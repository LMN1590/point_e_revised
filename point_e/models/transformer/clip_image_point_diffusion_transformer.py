import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Dict, Any, Iterable, Union
from PIL import Image
import math

from .point_diffusion_transformer import PointDiffusionTransformer
from ..pretrained_clip import ImageCLIP, FrozenImageCLIP
from ..blocks.timestep_embedding import timestep_embedding
from ...utils.const import ImageType

class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self, *,
        device:torch.device, dtype:torch.dtype,
        
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        n_ctx:int = 1024,
        **kwargs
    ):
        super().__init__(
            device=device, dtype=dtype, 
            n_ctx=n_ctx + int(token_cond), **kwargs
        )
        self.n_ctx = n_ctx
        self.token_cond = token_cond
        self.clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(device, cache_dir=cache_dir)
        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )
        self.cond_drop_prob = cond_drop_prob
        
    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs))
    
    def forward(
        self, x:torch.Tensor, t:torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
        **model_kwargs
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param texts: a batch of texts to condition on.
        :param embeddings: a batch of CLIP embeddings to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        clip_out = self.clip(batch_size=len(x), images=images, texts=texts, embeddings=embeddings)
        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        # Rescale the features to have unit variance
        clip_out = math.sqrt(clip_out.shape[1]) * clip_out

        clip_embed = self.clip_embed(clip_out)

        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]
        return self._forward_with_cond(x, cond)