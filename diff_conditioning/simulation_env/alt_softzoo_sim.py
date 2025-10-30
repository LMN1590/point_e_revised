import taichi as ti

import numpy as np
import torch

import random
from yaml import safe_load
import os
import shutil
import json
from typing import Dict, List, Tuple,Union
from tqdm import tqdm,trange
import json

from softzoo.configs.config_dataclass import FullConfig
from softzoo.envs import ENV_CONFIGS_DIR
from softzoo.utils.logger import Logger
from softzoo.utils.general_utils import save_pcd_to_mesh
from .utils.sim_utils import read_fixed_velocity

from point_e.diffusion.gaussian_diffusion import GaussianDiffusion

from .env import make_env
from .loss import make_loss
from .sap import CustomSAP
from . import controllers as controller_module

from .utils.path import CONFIG_DIR,DEFAULT_CFG_DIR
from ..base_cond import BaseCond
from .designer.encoded_finger.design import EncodedFinger
from .controllers.custom_finger_rep import CustomFingerRepController

from sap.config_dataclass import SAPConfig
from config.config_dataclass import ConditioningConfig
from logger import TENSORBOARD_LOGGER as tensorboard_logger,CSVLOGGER
import logging

class AltSoftzooSimulation(BaseCond):
    # region Initialization
    def __init__(
        self,
        config:FullConfig, sap_config:SAPConfig,
        name:str,
        grad_scale:float, calc_gradient:bool = False,
        grad_clamp:float = 1e-2,
        logging_bool:bool = True
    ):
        super(AltSoftzooSimulation, self).__init__(name,grad_scale,calc_gradient,grad_clamp,logging_bool)
        self.sap = CustomSAP(
            sap_config,
            device = torch.device(sap_config['device']),
            num_fingers=config.num_fingers
        )
        self.config = config
        self.torch_device = config.design_device
        
        ti.init(arch=ti.cuda, device_memory_fraction=config.device_memory_fraction, random_seed = config.seed)
        
        if config.eval:
            config.optimize_controller = False
            config.optimize_designer = False
        
        self._init_sim(config)
        
        self.post_substep_grad_fn = []
        if config.action_space == 'actuator_v':
            # only suppport particle-based representation for now
            self.env.design_space.instantiate_v_buffer()
            self.post_substep_grad_fn.append(self.env.design_space.add_v_with_buffer.grad)
            
        self.traj = None
        if 'TrajectoryFollowingLoss' in config.loss_types:
            traj_len = int(config.goal[-1])
            self.traj = ti.Vector.field(3, self.env.renderer.f_dtype, shape=(traj_len))
            traj_loss = self.loss_set.losses[config.loss_types.index('TrajectoryFollowingLoss')]
            traj_loss.reset()
            self.traj.from_numpy(traj_loss.data['traj'].to_numpy().astype(self.env.renderer.f_dtype_np)[:traj_len])
            
        self._init_checkpointing(config)
        self.data_for_plots = dict(reward=[], loss=[])

    def _init_sim(self,config:FullConfig):
        with open("diff_conditioning/simulation_env/designer/encoded_finger/config/base_config.json") as f:
            content = json.load(f)
        self.env = make_env(config) 
        # Define loss
        self.loss_set = make_loss(config, self.env, self.torch_device)
            
        self.controller = CustomFingerRepController(
            base_config=content,
            env=self.env,
            n_actuators=self.env.sim.solver.n_actuators,
            device = self.torch_device,
            active = config.active
        )
        
        self.designer = EncodedFinger(
            base_config = content,
            env = self.env,
            device = self.torch_device,
            bounding_box=self.config.gen_pointe_bounding_box
        )
        
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
    def load_config(cls,cfg_item:Union[str,Dict])->FullConfig:
        if isinstance(cfg_item,str):
            with open(cfg_item) as f:
                cfg = safe_load(f)
            return cls.load_config(cfg)
        else:
            default_cfg = cls._load_default_config()
            return FullConfig(**(default_cfg|cfg_item)) 
        
    # endregion
    
    def _init_checkpointing(self,config:FullConfig):
        # region Checkpointing
        ckpt_root_dir = os.path.join(config.out_dir, 'ckpt')
        os.makedirs(ckpt_root_dir, exist_ok=True)
        self.ckpt_dir_controller = None
        if config.optimize_designer or config.save_designer:
            self.design_dir = os.path.join(config.out_dir, 'design')
            os.makedirs(self.design_dir, exist_ok=True)

        with open(os.path.join(ckpt_root_dir, 'args.json'), 'w') as fp:
            json.dump(vars(config), fp, indent=4, sort_keys=True)

        env_config_path = os.path.join(ENV_CONFIGS_DIR, config.env_config_file)
        shutil.copy(env_config_path, os.path.join(ckpt_root_dir, 'env_config.yaml'))
        # endregion 
    # endregion
    
    # region Gradient Calc Wrapper
    def calculate_gradient(
        self, 
        ctrl_tensor:torch.Tensor,
        end_prob_mask:torch.Tensor,
        iter:int
    ) -> torch.Tensor:
        # x is shaped (B*2, C, N)
        x = ctrl_tensor
        B = x.shape[0]
        cur_loss = []
        sap_loss_lst = []
        accum_grad = torch.zeros_like(x)
        with torch.enable_grad():
            ep_reward,reward_log,design_loss = self.forward_sim(
                ctrl_tensor,
                end_prob_mask,
                iter,0,0
            ) # gripper: cpu(design) -> cuda:0 (env)
            all_loss,grad,grad_name_control = self.backward_sim()
            cur_loss.append(all_loss[-1])
            
            if self.calc_gradient:
                design_loss.backward(retain_graph=True)
                self.designer.out_cache['geometry'].backward(gradient=grad[None]['self.env.design_space.buffer.geometry'],retain_graph=True)
                self.designer.out_cache['softness'].backward(gradient=grad[None]['self.env.design_space.buffer.softness'],retain_graph=True)
                grad_control = [grad[s]['self.env.sim.solver.act_buffer'] for s in self.controller.all_s]
                self.controller.backward(grad_control)
                # print(ctrl_tensor.grad.reshape(-1).norm(2))
            cur_loss = np.array(cur_loss)
            tensorboard_logger.log_scalar("Simulation_SoftZoo/All_Batch_SoftZoo_Loss",cur_loss.mean())
    # endregion
    
    # region Simulation
    def forward_sim(
        self,
        ctrl_tensor:torch.Tensor, end_prob_mask:torch.Tensor,
        batch_idx:int,
        sampling_step:int,
        local_iter:int
    ):
        self.designer.reset()
        designer_out,design_loss = self.designer(ctrl_tensor,end_prob_mask)
        design = dict()
        for design_type in self.config.set_design_types:
            if design_type == 'actuator_direction': assert getattr(self.designer,'has_actuator_direction',False)
            design[design_type] = designer_out[design_type]
        obs = self.env.reset(
            design, batch_idx,
            sampling_step,local_iter,
            save_cur_iter = sampling_step%self.config.render_every_iter==0
        )
        self.controller.reset()
        ep_reward = 0.
        
        loss_reset_kwargs = {k: {} for k in self.config.loss_types}
        self.loss_set.reset(loss_reset_kwargs)
        
        if self.config.optimize_designer and (sampling_step%self.config.render_every_iter==0):
            if 'particle_based_representation' in str(self.env.design_space):
                for design_type in ['geometry','actuator','softness']:
                    design_fpath = os.path.join(self.design_dir, f'{design_type}_Batch_{batch_idx}_Sampling_{sampling_step:04d}_Local_{local_iter:04d}.pcd')
                    design_pcd = self.designer.save_pcd(design_fpath, design, design_type)
                save_pcd_to_mesh(
                    os.path.join(self.design_dir, f'mesh_Batch_{batch_idx}_Sampling_{sampling_step:04d}_Local_{local_iter:04d}.ply'), 
                    design_pcd
                )
                self.designer.save_actuator_direction(
                    design,
                    os.path.join(self.design_dir, f'actuator_direction_Batch_{batch_idx}_Sampling_{sampling_step:04d}_Local_{local_iter:04d}.npy'), 
                )
            elif 'voxel_based_representation' in str(self.env.design_space):
                for design_type in self.config.optimize_design_types:
                    design_fpath = os.path.join(self.design_dir, f'{design_type}_{sampling_step:04d}.ply')
                    self.designer.save_voxel_grid(design_fpath, design, design_type)
            else:
                raise NotImplementedError
            
        velocities_by_frame = read_fixed_velocity(self.config.fixed_v,self.config.n_frames)
        
        fixed_v = [0.,0.,0.]
        cur_v_idx = 0
        
        # self.env.sim.solver.set_gravity([0.,1.,0.])
        
        pbar = trange(0,self.config.n_frames,desc="Simulating",unit=' frames',position=2,leave=False)
        reward_log = []
        for frame in pbar:
            if frame >= velocities_by_frame[cur_v_idx][0]:
                fixed_v = velocities_by_frame[cur_v_idx][1]
                cur_v_idx +=1
            if self.config.custom_gravity:
                if frame == 10:self.env.sim.solver.set_gravity((0.,15.,0.))
                elif frame == 25:self.env.sim.solver.set_gravity((0.,0.,0.))

            current_s = self.env.sim.solver.current_s
            current_s_local = self.env.sim.solver.get_cyclic_s(current_s)
            act = self.controller(current_s, obs,ctrl_tensor)
            if self.config.action_space == 'particle_v':
                self.env.design_space.add_to_v(current_s_local, act) # only add v to the first local substep since v accumulates
                obs, reward, done, info = self.env.step(None,fixed_v)
            elif self.config.action_space == 'actuator_v':
                self.env.design_space.set_v_buffer(current_s, act)
                self.env.design_space.add_v_with_buffer(current_s, current_s_local)
                obs, reward, done, info = self.env.step(None,fixed_v)
            else:
                obs, reward, done, info = self.env.step(act,fixed_v)
            pbar.set_postfix({
                "reward": reward
            })
            reward_log.append(reward)
            ep_reward += reward

            if self.env.has_renderer and (sampling_step % self.config.render_every_iter == 0):
                if 'TrajectoryFollowingLoss' in self.config.loss_types: # plot trajectory
                    self.env.renderer.scene.particles(self.traj, radius=0.003)
                if hasattr(self.env.objective, 'render'):
                    self.env.objective.render()
                self.env.render()

            if done: break
        return ep_reward,reward_log,design_loss
    def backward_sim(self):
        grad_names:Dict[int,List] = dict()
        if self.config.action_space == 'particle_v':
            grad_name_control = 'self.env.sim.solver.v'
        elif self.config.action_space == 'actuator_v':
            grad_name_control = 'self.env.design_space.v_buffer'
        else:
            grad_name_control = 'self.env.sim.solver.act_buffer'
        if self.config.optimize_controller:
            for s in self.controller.all_s:
                if s not in grad_names.keys():
                    grad_names[s] = []
                grad_names[s].append(grad_name_control)
        if self.config.optimize_designer:
            s = None
            if s not in grad_names.keys():
                grad_names[s] = []
            for dsr_buffer_name in self.config.optimize_design_types:
                grad_names[s].append(f'self.env.design_space.buffer.{dsr_buffer_name}')
        try:
            all_loss, grad = self.loss_set.compute_loss(self.post_substep_grad_fn, compute_grad=len(grad_names) > 0, grad_names=grad_names)
        except Exception as e: # HACK
            raise e
            all_loss = np.zeros([1])
            grad = dict()
        return all_loss,grad,grad_name_control
    # endregion 
    @classmethod
    def init_cond(cls,config:ConditioningConfig,softzoo_config:FullConfig, sap_config:SAPConfig,**kwargs)->'AltSoftzooSimulation':
        return cls(
            name = config['name'],
            config = softzoo_config,
            sap_config = sap_config,
            grad_scale = config['grad_scale'],
            grad_clamp = config['grad_clamp'],
            calc_gradient = config['calc_gradient'],
            logging_bool = config['logging_bool']
        )

# if __name__=='__main__':
#     import numpy as np
#     import open3d as o3d
    
#     full_config,sap_config = SoftzooSimulation.load_config(
#         cfg_path = 'config/softzoo_config.yaml',
#         sap_cfg_path = 'config/sap_config.yaml'
#     )
#     cond_cls = SoftzooSimulation(
#         config = full_config,
#         sap_config= sap_config,
#         grad_scale=1e-1,
#         calc_gradient=True,
#         grad_clamp=1e-2
#     )
    
#     loaded_pcd = np.load('sample_generated_pcd.npz')
#     x = torch.from_numpy(loaded_pcd['coords']).detach().to(sap_config['device'])
#     # pcd = o3d.io.read_point_cloud('hand.pcd')
#     # x_hand=torch.from_numpy(np.array(pcd.points)).requires_grad_(True)

    
#     x = x.permute(1,0)
#     # x_hand = normalize_to_unit_box(x_hand).permute(1,0)
#     x = torch.stack([
#         x.detach().clone().requires_grad_(True),
#         x.detach().clone().requires_grad_(True),
#     ],dim=0)

#     scaled_grad = cond_cls.calculate_gradient(x,torch.tensor([0,0,0]))
#     print(scaled_grad)
    
    
