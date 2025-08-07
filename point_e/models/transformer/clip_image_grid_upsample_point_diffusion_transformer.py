import torch
import torch.nn as nn

from typing import Optional, Dict, Any, Iterable

from .upsample_point_diffusion_transformer import UpsamplePointDiffusionTransformer
from ..pretrained_clip import ImageCLIP, FrozenImageCLIP
from ...utils.const import ImageType
from ..blocks.timestep_embedding import timestep_embedding

class CLIPImageGridUpsamplePointDiffusionTransformer(UpsamplePointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096 - 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx

        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "images" not in model_kwargs:
            zero_emb = torch.zeros(
                [batch_size, self.clip.grid_feature_dim, self.clip.grid_size**2],
                device=next(self.parameters()).device,
            )
            return dict(embeddings=zero_emb, low_res=model_kwargs["low_res"])
        with torch.no_grad():
            return dict(
                embeddings=self.clip.embed_images_grid(model_kwargs["images"]),
                low_res=model_kwargs["low_res"],
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        low_res: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
        **model_kwargs
    ):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        elif embeddings is not None:
            clip_out = embeddings
        else:
            # Support unconditional generation.
            clip_out = torch.zeros(
                [len(x), self.clip.grid_feature_dim, self.clip.grid_size**2],
                dtype=x.dtype,
                device=x.device,
            )

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond), (clip_embed, True), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)
