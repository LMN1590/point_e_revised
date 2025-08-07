import taichi as ti

import numpy as np
import torch

import random
from yaml import safe_load
import os
import shutil
import json
from typing import Dict

from softzoo.configs.config_dataclass import FullConfig
from softzoo.envs import ENV_CONFIGS_DIR
from softzoo.utils.logger import Logger

from point_e.diffusion.gaussian_diffusion import GaussianDiffusion

from .env import make_env
from .loss import make_loss
from . import controllers as controller_module

from .utils.path import CONFIG_DIR,DEFAULT_CFG_DIR
from ..base_cond import BaseCond

class SoftzooSimulation(BaseCond):
    # region Initialization
    def __init__(self,config:FullConfig, grad_scale:float, calc_gradient:bool = False):
        super(SoftzooSimulation, self).__init__(grad_scale,calc_gradient)
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.torch_seed)
        
        ti.init(arch=ti.cuda, device_memory_fraction=config.device_memory_fraction, random_seed = config.seed)
        
        if config.eval:
            config.optimize_controller = False
            config.optimize_designer = False
        
        self._init_sim(config)
        
        post_substep_grad_fn = []
        if config.action_space == 'actuator_v':
            # only suppport particle-based representation for now
            self.env.design_space.instantiate_v_buffer()
            post_substep_grad_fn.append(self.env.design_space.add_v_with_buffer.grad)
            
        traj = None
        if 'TrajectoryFollowingLoss' in config.loss_types:
            traj_len = int(config.goal[-1])
            traj = ti.Vector.field(3, self.env.renderer.f_dtype, shape=(traj_len))
            traj_loss = self.loss_set.losses[config.loss_types.index('TrajectoryFollowingLoss')]
            traj_loss.reset()
            traj.from_numpy(traj_loss.data['traj'].to_numpy().astype(self.env.renderer.f_dtype_np)[:traj_len])
            
        self._init_checkpointing(config)
        
        logger = Logger(args.out_dir)
        def time_fn(func):
            def inner(*args, **kwargs):
                logger.tic(func.__qualname__)
                out = func(*args, **kwargs)
                logger.toc(func.__qualname__)
                return out
            return inner

        self.loss_set.compute_loss = time_fn(self.loss_set.compute_loss)
        self.controller.update = time_fn(self.controller.update)
        self.designer.update = time_fn(self.designer.update)
        
        self.data_for_plots = dict(reward=[], loss=[])

    def _init_sim(self,config:FullConfig):
        self.env = make_env(config) 
        torch_device = 'cuda' if config.non_taichi_device == 'torch_gpu' else 'cpu'

        # Define loss
        self.loss_set = make_loss(config, self.env, torch_device)
            
        self.controller = controller_module.make(config, self.env, torch_device)
        
    # region Load Config
    @classmethod
    def _load_default_config(cls):
        with open(os.path.join(DEFAULT_CFG_DIR,'default.yaml')) as f:
            default_cfg = safe_load(f)
        with open(os.path.join(DEFAULT_CFG_DIR,'controller.yaml')) as f:
            controller_cfg = safe_load(f)
        with open(os.path.join(DEFAULT_CFG_DIR,'designer.yaml')) as f:
            designer_cfg = safe_load(f)
            
        return default_cfg|controller_cfg|designer_cfg
    
    @classmethod
    def load_config(cls,cfg_path:str):
        with open(os.path.join(CONFIG_DIR,cfg_path)) as f:
            cfg = safe_load(f)
        
        default_cfg = cls._load_default_config()
        
        return FullConfig(**(default_cfg|cfg))
    # endregion
    
    def _init_checkpointing(self,config:FullConfig):
        # region Checkpointing
        ckpt_root_dir = os.path.join(config.out_dir, 'ckpt')
        os.makedirs(ckpt_root_dir, exist_ok=True)
        design_dir = None
        ckpt_dir_designer = None
        ckpt_dir_controller = None
        if config.optimize_designer or config.save_designer:
            design_dir = os.path.join(config.out_dir, 'design')
            os.makedirs(design_dir, exist_ok=True)

            ckpt_dir_designer = os.path.join(ckpt_root_dir, 'designer')
            os.makedirs(ckpt_dir_designer, exist_ok=True)

        if config.optimize_controller or config.save_controller:
            ckpt_dir_controller = os.path.join(ckpt_root_dir, 'controller')
            os.makedirs(ckpt_dir_controller, exist_ok=True)

        with open(os.path.join(ckpt_root_dir, 'args.json'), 'w') as fp:
            json.dump(vars(config), fp, indent=4, sort_keys=True)

        env_config_path = os.path.join(ENV_CONFIGS_DIR, config.env_config_file)
        shutil.copy(env_config_path, os.path.join(ckpt_root_dir, 'env_config.yaml'))
        # endregion 
    # endregion
    
    def calculate_loss(
        self, 
        x: torch.Tensor, t: torch.Tensor,
        p_mean_var:Dict[str,torch.Tensor],
        diffusion:GaussianDiffusion, 
        **model_kwargs
    ) -> torch.Tensor:
        
        if 'original_ts' in model_kwargs:
            t = model_kwargs['original_ts']
        
        pred_xstart = diffusion._predict_xstart_from_eps(
            x,t,
            p_mean_var['eps']
        )
        

if __name__=='__main__':
    print(SoftzooSimulation.load_config('custom_cfg.yaml'))