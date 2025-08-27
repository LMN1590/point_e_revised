import torch
import numpy as np

import trimesh
import open3d as o3d

from tqdm import trange,tqdm
import os

from sap.config_dataclass import SAPConfig
from sap.utils.schedule_utils import StepLearningRateSchedule,adjust_learning_rate
from sap.utils.optimizer_utils import update_optimizer
from sap.utils.mesh_pc_utils import sample_pc_in_mesh
from sap.utils.gradient_utils import gaussian_kernel
from sap.optimization import Trainer

class CustomSAP:
    def __init__(self,config:SAPConfig,device:torch.device):
        self.sap_config = config
        self.device = device
    
    def dense_sample(self,x_0:torch.Tensor,iter_idx:int,batch_idx:int,save_res:bool):
        # x_0: (N,3)
        data = self._preprocess(x_0)
        inputs = self._prepare_input()
        
        if self.sap_config['train']['exp_mesh'] and save_res:
            os.makedirs(self.sap_config['train']['dir_mesh'],exist_ok=True)
        if self.sap_config['train']['exp_pcl'] and save_res:
            os.makedirs(self.sap_config['train']['dir_pcl'],exist_ok=True)
        
        input_scheduler = StepLearningRateSchedule(
            initial = self.sap_config['train']['schedule']['initial'],
            interval= self.sap_config['train']['schedule']['interval'],
            factor = self.sap_config['train']['schedule']['factor'],
            final = self.sap_config['train']['schedule']['final']
        )
        
        optimizer = update_optimizer(inputs,input_scheduler,epoch = 0)
        start_epoch = -1
        trainer = Trainer(self.sap_config,optimizer,device=self.device)
        
        pbar = trange(start_epoch+1, self.sap_config['train']['total_epochs']+1,desc="Training",unit=' epoch')
        for epoch in pbar:
            # schedule the learning rate
            if epoch>0:
                if (epoch % input_scheduler.interval == 0):
                    adjust_learning_rate(input_scheduler, optimizer, epoch)
                    # print('[epoch {}] adjust pcl_lr to: {}'.format(epoch, input_scheduler.get_learning_rate(epoch)))
                    tqdm.write(f'[epoch {epoch}] adjust pcl_lr to: {input_scheduler.get_learning_rate(epoch):.6f}')
            
            loss, loss_each = trainer.train_step(data, inputs, None, epoch)
            
            metrics = {"loss": f"{loss:.5f}"}
            if loss_each is not None:
                for k, l in loss_each.items():
                    if l.item() != 0.: metrics[f"loss_{k}"] = f"{l.item():.5f}"
            pbar.set_postfix(metrics)
            
            # resample and gradually add new points to the source pcl
            if  (epoch > 0) & \
                (self.sap_config['train']['resample_every']!=0) & \
                (epoch % self.sap_config['train']['resample_every'] == 0) & \
                (epoch < self.sap_config['train']['total_epochs']):
                        inputs = trainer.point_resampling(inputs)
                        optimizer = update_optimizer(inputs,epoch=epoch, schedule=input_scheduler)
                        trainer = Trainer(self.sap_config, optimizer, device=self.device)

        mesh = trainer.export_mesh(inputs,data['center'].cpu().numpy(), data['scale'].cpu().numpy()*(1/0.9)) 
        if self.sap_config['train']['exp_mesh'] and save_res:
            o3d.io.write_triangle_mesh(
                os.path.join(self.sap_config['train']['dir_mesh'],f'mesh_{iter_idx}_{batch_idx}.ply'),
                mesh
            )
        pcd = sample_pc_in_mesh(
            mesh, num_points=self.sap_config['sample']['num_points'],
            density=self.sap_config['sample']['density'], 
            voxel_size=self.sap_config['sample']['voxel_size']
        )
        if self.sap_config['train']['exp_pcl'] and save_res:
            o3d.io.write_point_cloud(
                os.path.join(self.sap_config['train']['dir_pcl'],f'pcd_{iter_idx}_{batch_idx}.ply'),
                pcd
            )
        return torch.from_numpy(np.array(pcd.points)).float().to(self.device)
        
    def _preprocess(self,points:torch.Tensor):
        # points: (N,3)
        center = points.mean(dim=0) # (3,)
        scale = torch.max(torch.max(torch.abs(points-center),dim=0).values)
        
        vertices = points-center
        vertices /= scale
        vertices *= 0.9
        
        target_pts = vertices.to(self.device)[None].float()
        
        return {
            "target_points": target_pts,
            "center": center,
            "scale":scale
        }
        
    def _prepare_input(self):
        sphere_radius = self.sap_config['model']['sphere_radius']
        sphere_mesh = trimesh.creation.uv_sphere(radius=sphere_radius, count=[256,256])
        points, idx = sphere_mesh.sample(self.sap_config['model']['num_points'], return_index=True)
        points += 0.5 # make sure the points are within the range of [0, 1)
        normals = sphere_mesh.face_normals[idx]
        points = torch.from_numpy(points).unsqueeze(0).to(self.device)
        normals = torch.from_numpy(normals).unsqueeze(0).to(self.device)
        
        points = torch.log(points/(1-points)) # inverse sigmoid
        inputs = torch.cat([points, normals], dim=-1).float()
        inputs.requires_grad = True
        
        return inputs
    
    def calculate_x0_grad(
        self,
        dense_gripper:torch.Tensor, dense_gradients:torch.Tensor,
        surface_gripper:torch.Tensor,
    ):
        assert dense_gripper.shape == dense_gradients.shape
        gradient_matrix = gaussian_kernel(surface_gripper, dense_gripper, alpha=self.sap_config['gradient_alpha'])
        return gradient_matrix @ dense_gradients
        