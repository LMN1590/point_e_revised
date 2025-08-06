import torch

from typing import Dict

from .base_cond import BaseCond
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion

class OriginDistanceCond(BaseCond):
    def calculate_loss(
        self, 
        x: torch.Tensor, t: torch.Tensor,
        p_mean_var:Dict[str,torch.Tensor],
        diffusion:GaussianDiffusion, 
        **model_kwargs
    ) -> torch.Tensor:
        
        pred_xstart = diffusion._predict_xstart_from_eps(
            x,t,
            p_mean_var['eps']
        )
        B = pred_xstart.shape[0]
        pos = pred_xstart[:B//2,:3]
        return (pos**2).mean()