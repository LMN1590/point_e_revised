import torch
import numpy as np

from typing import Dict

from config.config_dataclass import ConditioningConfig
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion
from logger import TENSORBOARD_LOGGER,CSVLOGGER

class BaseCond:
    def __init__(
        self, name:str, 
        grad_scale:float, calc_gradient:bool = False, grad_clamp:float = 1e-2,
        logging_bool:bool=True,
        **kwargs
    ):
        self.name = name
        self.calc_gradient = calc_gradient
        self.grad_scale = grad_scale
        self.grad_clamp = grad_clamp
        self.logging_bool = logging_bool
        self.loss_lst = []
        self.grad_lst = []
    
    def calculate_gradient(
        self, 
        x: torch.Tensor, t: torch.Tensor,
        p_mean_var:Dict[str,torch.Tensor],
        diffusion:GaussianDiffusion, 
        local_iter:int, 
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
                loss = self.calculate_loss(pos[i],t,p_mean_var,diffusion,local_iter,**model_kwargs)
                
                if self.calc_gradient:
                    cur_grad = torch.autograd.grad(loss, x)[0]
                    accum_grad[i,:3] = cur_grad[i,:3]
                
                if self.logging_bool:
                    CSVLOGGER.log({
                        "phase": self.name,
                        
                        "sampling_step": t_sample,
                        "local_iter": local_iter,
                        "batch_idx": i,
                        
                        'loss': loss.item(),
                        'grad_norm':0. if not self.calc_gradient else cur_grad.norm(2).item()
                    })
                if torch.is_tensor(loss):
                    loss = loss.item()
                cur_loss.append(loss)

        cur_loss = np.array(cur_loss)
        self.grad_lst.append(accum_grad)   
        self.loss_lst.append(cur_loss)
        scaled_gradient = torch.clamp(-accum_grad*self.grad_scale,min = -self.grad_clamp,max=self.grad_clamp)

        if self.logging_bool:
            TENSORBOARD_LOGGER.log_scalar(f"{self.name}/All_Batch_Loss",cur_loss.mean())
            TENSORBOARD_LOGGER.log_scalar(f"{self.name}/All_Batch_GradientNorm",scaled_gradient.reshape(-1).norm(2))
            # TENSORBOARD_LOGGER.increment_step()
            
            CSVLOGGER.log({
                "phase": f"{self.name}_Overall",
                
                "sampling_step": t.tolist()[0],
                "local_iter": local_iter,
                
                "loss": cur_loss.mean(),
                "grad_norm":scaled_gradient.reshape(-1).norm(2).item()
            })    
        return scaled_gradient   # negative sign: push mean back toward origin
        
    def calculate_loss(self, x:torch.Tensor,t:torch.Tensor,p_mean_var:Dict[str,torch.Tensor],diffusion:GaussianDiffusion,local_iter:int, **model_kwargs) -> torch.Tensor:
        raise NotImplementedError("Base Loss Calculation called")
    
    @classmethod
    def init_cond(cls,config:ConditioningConfig,**kwargs)->'BaseCond':
        return cls(
            name = config['name'],
            grad_scale = config['grad_scale'],
            grad_clamp = config['grad_clamp'],
            calc_gradient = config['calc_gradient'],
            logging_bool = config['logging_bool']
        )