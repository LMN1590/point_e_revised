import torch

from typing import Dict

from .base_cond import BaseCond
from custom_diffusion.diffusion.gaussian_diffusion import GaussianDiffusion

class OriginDistanceCond(BaseCond):
    def calculate_loss(
        self, pos:torch.Tensor,t:torch.Tensor,
        p_mean_var:Dict[str,torch.Tensor],diffusion:GaussianDiffusion, 
        local_iter:int,
        **model_kwargs
    ) -> torch.Tensor:
        # pos: (3,N)
        return (pos**2).mean()