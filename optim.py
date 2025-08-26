import torch
import numpy as np

from plyfile import PlyData
import trimesh
import open3d as o3d

from typing import Dict,Literal
import yaml
from tqdm import trange,tqdm
import os

from config_dataclass import SAPConfig
from src.utils.schedule_utils import StepLearningRateSchedule,adjust_learning_rate
from src.utils.optimizer_utils import update_optimizer
from src.utils.logger_utils import initialize_logger
from src.utils.mesh_pc_utils import sample_pc_in_mesh
from src.optimization import Trainer

def load_config(config_path:str)->SAPConfig:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def load_ply(ply_path:str,device:torch.device)->Dict[Literal['target_points','center','scale','original_points'],torch.Tensor]:
    plydata = PlyData.read(ply_path)
    org_vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    N = org_vertices.shape[0]
    center = org_vertices.mean(0)
    scale = np.max(np.max(np.abs(org_vertices - center), axis=0))
    vertices = org_vertices-center
    vertices /= scale
    vertices *= 0.9

    src_pts=torch.tensor(org_vertices,device=device)[None].float()
    target_pts = torch.tensor(vertices, device=device)[None].float()
    return {
        "original_points":src_pts,
        "target_points": target_pts,
        "center": torch.from_numpy(center) if not torch.is_tensor(center) else center,
        "scale":torch.from_numpy(np.array([scale])) if not torch.is_tensor(scale) else scale
    }

def optimize(config:SAPConfig, data:Dict[Literal['target_points','center','scale','original_points'],torch.Tensor]):
    # region Init Point Cloud
    sphere_radius = config['model']['sphere_radius']
    sphere_mesh = trimesh.creation.uv_sphere(radius=sphere_radius, count=[256,256])
    points, idx = sphere_mesh.sample(config['model']['num_points'], return_index=True)
    points += 0.5 # make sure the points are within the range of [0, 1)
    normals = sphere_mesh.face_normals[idx]
    points = torch.from_numpy(points).unsqueeze(0).to(device)
    normals = torch.from_numpy(normals).unsqueeze(0).to(device)
    
    points = torch.log(points/(1-points)) # inverse sigmoid
    inputs = torch.cat([points, normals], dim=-1).float()
    inputs.requires_grad = True
    # endregion
    
    initialize_logger(config)
    
    input_scheduler = StepLearningRateSchedule(
        initial = config['train']['schedule']['initial'],
        interval= config['train']['schedule']['interval'],
        factor = config['train']['schedule']['factor'],
        final = config['train']['schedule']['final']
    )
    optimizer = update_optimizer(inputs,input_scheduler,epoch = 0)
    start_epoch = -1
    trainer = Trainer(config,optimizer,device=device)
    
    # training loop
    pbar = trange(start_epoch+1, config['train']['total_epochs']+1,desc="Training",unit=' epoch')
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
        if (epoch > 0) & \
           (config['train']['resample_every']!=0) & \
           (epoch % config['train']['resample_every'] == 0) & \
           (epoch < config['train']['total_epochs']):
                inputs = trainer.point_resampling(inputs)
                optimizer = update_optimizer(inputs,epoch=epoch, schedule=input_scheduler)
                trainer = Trainer(config, optimizer, device=device)

    mesh = trainer.export_mesh(inputs,data['center'].cpu().numpy(), data['scale'].cpu().numpy()*(1/0.9)) 
    return mesh
    
def gaussian_kernel(surface_pc:torch.Tensor, dense_pc:torch.Tensor,alpha:float = 20.):
    dists = torch.cdist(surface_pc, dense_pc, p=2)**2
    K = torch.exp(-alpha * dists)
    return K / (K.sum(dim=1, keepdim=True) + 1e-9)
    
if __name__ == "__main__":
    config = load_config('config.yaml')
    ply_path = 'data/hand.ply'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_ply(ply_path,device) # B,N,C
    
    mesh = optimize(config,data)
    outdir_mesh = os.path.join(config['train']['out_dir'], 'final_mesh.ply')
    o3d.io.write_triangle_mesh(outdir_mesh, mesh)
    
    pcd = sample_pc_in_mesh(mesh, num_points=config['sample']['num_points'], density=config['sample']['density'], voxel_size=config['sample']['voxel_size'])
    outdir_pcd = os.path.join(config['train']['out_dir'], 'final_pcd.ply')
    o3d.io.write_point_cloud(outdir_pcd, pcd)
    
    res_pts = torch.from_numpy(np.array(pcd.points)).to(device).to(torch.float64)
    src_pts = data['original_points'][0].to(torch.float64)
    
    # Convert to numpy (on CPU)
    res_pts_np = res_pts.detach().cpu().numpy()
    src_pts_np = src_pts.detach().cpu().numpy()
    
    # Save as npz
    outdir_npz = os.path.join(config['train']['out_dir'], 'final_points.npz')
    np.savez(outdir_npz, res_pts=res_pts_np, src_pts=src_pts_np)
    
    print(f"Saved points to {outdir_npz}")