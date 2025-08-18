import torch

from typing import Dict

from .base_cond import BaseCond
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion

class OriginDistanceCond(BaseCond):
    def calculate_loss(
        self, pos:torch.Tensor,t:torch.Tensor,p_mean_var:Dict[str,torch.Tensor],diffusion:GaussianDiffusion, **model_kwargs
    ) -> torch.Tensor:

        return (pos**2).mean()