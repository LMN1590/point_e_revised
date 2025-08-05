import torch

from .base_cond import BaseCond

class OriginDistanceCond(BaseCond):
    def calculate_loss(self, x: torch.Tensor, t: torch.Tensor, pred_xstart:torch.Tensor, **model_kwargs) -> torch.Tensor:
        pos = pred_xstart[:,:3]
        return (pos**2).sum()