import torch

from typing import Dict

from point_e.diffusion.gaussian_diffusion import GaussianDiffusion
from point_e.diffusion.diff_utils import _extract_into_tensor

class BaseCond:
    def __init__(self, grad_scale:float, calc_gradient:bool = False, grad_clamp:float = 1e-2,*args):
        self.calc_gradient = calc_gradient
        self.grad_scale = grad_scale
        self.grad_clamp = grad_clamp
        self.loss_lst = []
        self.grad_lst = []
    
    def calculate_gradient(
        self, 
        x: torch.Tensor, t: torch.Tensor,
        p_mean_var:Dict[str,torch.Tensor],
        diffusion:GaussianDiffusion, 
        **model_kwargs
    ):
        # x is shaped (B*2, C, N)
        
        x = x.detach().requires_grad_(True)
        B = x.shape[0]
        cur_loss = []
        accum_grad = torch.zeros_like(x)
        with torch.enable_grad():
            if 'original_ts' in model_kwargs:
                t = model_kwargs['original_ts']
            
            pred_xstart = diffusion._predict_xstart_from_eps(
                x,t,
                p_mean_var['eps']
            )
            pos = pred_xstart[:B//2,:3]
            
            for i,t_sample in zip(range(B//2),t.tolist()):
                loss = self.calculate_loss(pos[i],t,p_mean_var,diffusion,**model_kwargs)
                cur_loss.append(loss)
                
                if self.calc_gradient:
                    grad = torch.autograd.grad(loss, x)[0]
                    accum_grad[i,:3] = grad[i,:3]
                    accum_grad[i+B//2,:3] = grad[i,:3]
        self.grad_lst.append(accum_grad)   
        self.loss_lst.append(cur_loss)

        return -accum_grad*self.grad_scale  # negative sign: push mean back toward origin
        
    def calculate_loss(self, x:torch.Tensor,t:torch.Tensor,p_mean_var:Dict[str,torch.Tensor],diffusion:GaussianDiffusion, **model_kwargs) -> torch.Tensor:
        raise NotImplementedError("Base Loss Calculation called")