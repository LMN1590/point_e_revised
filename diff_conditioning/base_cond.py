import torch

from point_e.diffusion.gaussian_diffusion import GaussianDiffusion
from point_e.diffusion.diff_utils import _extract_into_tensor

class BaseCond:
    def __init__(self, grad_scale:float, calc_gradient:bool = False,*args):
        self.calc_gradient = calc_gradient
        self.grad_scale = grad_scale
        self.loss_lst = []
        self.grad_lst = []
    
    def calculate_gradient(self, x:torch.Tensor,t:torch.Tensor, **model_kwargs):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = self.calculate_loss(x,t,**model_kwargs)
        # print(loss.requires_grad, loss.grad_fn)
        self.loss_lst.append(loss)
    

        grad = torch.autograd.grad(loss, x)[0]  # gradient wrt x
        if not self.calc_gradient:
            grad = torch.zeros_like(grad)
        self.grad_lst.append(grad)
        return -grad*self.grad_scale  # negative sign: push mean back toward origin
        
    def calculate_loss(self, x:torch.Tensor,t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        raise NotImplementedError("Base Loss Calculation called")
    
    def _predict_xstart_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor,diffusion_param:GaussianDiffusion):
        assert eps is not None, "Epsilon is None, model_mean_type is potentially not epsilon"
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(diffusion_param.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(diffusion_param.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )