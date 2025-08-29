import torch

from typing import List, Dict,Callable

from config.config_dataclass import ConditioningConfig
from point_e.diffusion.gaussian_diffusion import GaussianDiffusion

from .base_cond import BaseCond
from .simulation_env import SoftzooSimulation
from .cond_dist_to_origin import OriginDistanceCond

from logger import TENSORBOARD_LOGGER,CSVLOGGER

COND_CLS:Dict[str,Callable[...,BaseCond]] = {
    "Dist_To_Origin": OriginDistanceCond.init_cond,
    "SoftZoo_Sim": SoftzooSimulation.init_cond
}

class CondSet:
    def __init__(self, cond_config_lst:List[ConditioningConfig],cond_overall_logging:bool=True,**kwargs):
        self.cond_overall_logging = cond_overall_logging
        self.cond_config_lst = cond_config_lst
        self.cond_cls:List[BaseCond] = []
        for config in cond_config_lst:
            name = config['name']
            cls_method = COND_CLS[name]
            self.cond_cls.append(cls_method(config,**kwargs))
            
        
        
    def calculate_gradient(
        self, 
        x: torch.Tensor, t: torch.Tensor,
        p_mean_var:Dict[str,torch.Tensor],
        diffusion:GaussianDiffusion, 
        local_iter:int, 
        **model_kwargs
    )->torch.Tensor:
        if 'original_ts' in model_kwargs:
            t = model_kwargs['original_ts']
        
        accum_grad = torch.zeros_like(x)
        for cond_cls in self.cond_cls:
            accum_grad += cond_cls.calculate_gradient(
                x= x, t = t,
                p_mean_var = p_mean_var,
                diffusion = diffusion,
                local_iter=local_iter,
                **model_kwargs
            )
        if self.cond_overall_logging:
            TENSORBOARD_LOGGER.log_scalar("Overall/All_Batch_GradientNorm",accum_grad.view(-1).norm(2))
            # tensorboard_logger.increment_step()
            
            CSVLOGGER.log({
                "phase": "Overall",
                
                "sampling_step": t.tolist()[0],
                "local_iter": local_iter,
                
                "grad_norm":accum_grad.view(-1).norm(2).item()
            })
        
        TENSORBOARD_LOGGER.increment_step()
        return accum_grad