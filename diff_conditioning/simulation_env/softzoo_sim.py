import taichi as ti

import numpy as np
import torch

import random
from yaml import safe_load
import os
import shutil
import json
from typing import Dict
from tqdm import tqdm

from softzoo.configs.config_dataclass import FullConfig
from softzoo.envs import ENV_CONFIGS_DIR
from softzoo.utils.logger import Logger
from softzoo.utils.general_utils import save_pcd_to_mesh
from .utils.sim_utils import read_fixed_velocity

from point_e.diffusion.gaussian_diffusion import GaussianDiffusion

from .env import make_env
from .loss import make_loss
from . import controllers as controller_module

from .utils.path import CONFIG_DIR,DEFAULT_CFG_DIR
from ..base_cond import BaseCond
from .designer.generated_pointe import GeneratedPointEPCD

class SoftzooSimulation(BaseCond):
    # region Initialization
    def __init__(self,config:FullConfig, grad_scale:float, calc_gradient:bool = False):
        super(SoftzooSimulation, self).__init__(grad_scale,calc_gradient)
        self.config = config
        self.torch_device = 'cuda' if config.non_taichi_device == 'torch_gpu' else 'cpu'
        
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
        
        logger = Logger(config.out_dir)
        def time_fn(func):
            def inner(*args, **kwargs):
                logger.tic(func.__qualname__)
                out = func(*args, **kwargs)
                logger.toc(func.__qualname__)
                return out
            return inner

        self.loss_set.compute_loss = time_fn(self.loss_set.compute_loss)
        self.controller.update = time_fn(self.controller.update)
        # self.designer.update = time_fn(self.designer.update)
        
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
    
    # region Gradient Calc Wrapper
    def calculate_gradient(self, x:torch.Tensor,t:torch.Tensor, **model_kwargs):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = self.calculate_loss(x,t,**model_kwargs)
        # print(loss.requires_grad, loss.grad_fn)
        self.loss_lst.append(loss)
    
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
        B = pred_xstart.shape[0]
        pos = pred_xstart[:B//2,:3]
        self.designer = GeneratedPointEPCD(
            lr = self.config.designer_lr,
            env = self.env,
            gripper_pos_tensor = pos,
            device = self.torch_device,
        )
        return torch.zeros((1,1))
    # endregion
    
    # # region Simulation
    # def forward_sim(
    #     self, it:int
    # ):
    #     self.designer.reset()
    #     designer_out = self.designer()
    #     design = dict()
    #     for design_type in self.config.set_design_types:
    #         if design_type == 'actuator_direction': assert getattr(self.designer,'has_actuator_direction',False)
    #         design[design_type] = designer_out[design_type]
            
    #     obs = self.env.reset(design)
    #     self.controller.reset()
    #     ep_reward = 0.
        
    #     if self.config.optimize_designer and (it%self.config.render_every_iter==0):
    #         if 'particle_based_representation' in str(self.env.design_space):
    #             for design_type in self.config.optimize_design_types:
    #                 design_fpath = os.path.join(design_dir, f'{design_type}_{it:04d}.pcd')
    #                 design_pcd = self.designer.save_pcd(design_fpath, design, design_type)
    #             save_pcd_to_mesh(os.path.join(design_dir, f'mesh_{it:04d}.ply'), design_pcd)
    #         elif 'voxel_based_representation' in str(self.env.design_space):
    #             for design_type in self.config.optimize_design_types:
    #                 design_fpath = os.path.join(design_dir, f'{design_type}_{it:04d}.ply')
    #                 self.designer.save_voxel_grid(design_fpath, design, design_type)
    #         else:
    #             raise NotImplementedError
            
    #     velocities_by_frame = read_fixed_velocity(self.config.fixed_v,self.config.n_frames)
        
    #     fixed_v = [0.,0.,0.]
    #     cur_v_idx = 0
        
    #     for frame in tqdm(range(self.config.n_frames), desc=f'Forward #{it:04d}'):
    #         if frame >= velocities_by_frame[cur_v_idx][0]:
    #             fixed_v = velocities_by_frame[cur_v_idx][1]
    #             cur_v_idx +=1
    #         # if frame == 0:
    #         #     env.sim.solver.set_gravity((0.,-9.81,0.))
    #         # elif frame == 125:
    #         #     env.sim.solver.set_gravity((0.,9.81,0.))
            
    #         current_s = self.env.sim.solver.current_s
    #         current_s_local = self.env.sim.solver.get_cyclic_s(current_s)
    #         act = self.controller(current_s, obs)
    #         if self.config.action_space == 'particle_v':
    #             self.env.design_space.add_to_v(current_s_local, act) # only add v to the first local substep since v accumulates
    #             obs, reward, done, info = self.env.step(None,fixed_v)
    #         elif self.config.action_space == 'actuator_v':
    #             self.env.design_space.set_v_buffer(current_s, act)
    #             self.env.design_space.add_v_with_buffer(current_s, current_s_local)
    #             obs, reward, done, info = self.env.step(None,fixed_v)
    #         else:
    #             obs, reward, done, info = env.step(act,fixed_v)
    #         ep_reward += reward

    #         if self.env.has_renderer and (it % self.config.render_every_iter == 0):
    #             if 'TrajectoryFollowingLoss' in args.loss_types: # plot trajectory
    #                 env.renderer.scene.particles(traj, radius=0.003)

    #             if hasattr(env.objective, 'render'):
    #                 env.objective.render()

    #             env.render()

    #         if (it % args.save_every_iter == 0):
    #             if args.optimize_designer or args.save_designer:
    #                 designer.save_checkpoint(os.path.join(ckpt_dir_designer, f'iter_{it:04d}.ckpt'))
    #             if args.optimize_controller or args.save_controller:
    #                 controller.save_checkpoint(os.path.join(ckpt_dir_controller, f'iter_{it:04d}.ckpt'))

    #         if done:
    #             break
    #     return ep_reward
        

if __name__=='__main__':
    config = SoftzooSimulation.load_config('custom_cfg.yaml')
    env = SoftzooSimulation(config,0.3,True)