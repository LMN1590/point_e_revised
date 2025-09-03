import torch
import numpy as np

import trimesh
import open3d as o3d
from kaolin.ops.mesh import check_sign

from typing import Optional, Dict,Literal
import os

from sap.config_dataclass import SAPConfig

from .model import PSR2Mesh
from .dpsr import DPSR
from .utils.mesh_pc_utils import mc_from_psr,verts_on_largest_mesh,export_pointcloud
from .utils.visualize_utils import visualize_psr_grid
from .loss import chamfer_distance_surface

class Trainer:
    '''
    Args:
        cfg       : config file
        optimizer : pytorch optimizer object
        device    : pytorch device
    '''
    def __init__(self, cfg:SAPConfig, optimizer:torch.optim.Optimizer, device:Optional[torch.device]=None):
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.psr2mesh = PSR2Mesh.apply
        self.data_type = cfg['data']['data_type']

        # initialize DPSR
        self.dpsr = DPSR(
            res=(
                cfg['model']['grid_res'], 
                cfg['model']['grid_res'], 
                cfg['model']['grid_res']
            ), 
            sig=cfg['model']['psr_sigma']
        )
        # if torch.cuda.device_count() > 1:    
        #     self.dpsr = torch.nn.DataParallel(self.dpsr) # parallell DPSR
        self.dpsr = self.dpsr.to(device)
    
    # region Training Step
    def train_step(self, data:Dict, inputs:torch.Tensor, model, it):
        ''' Performs a training step.

        Args:
            data (dict)              : data dictionary
            inputs (torch.tensor)    : input point clouds
            model (nn.Module or None): a neural network or None
            it (int)                 : the number of iterations
        '''

        self.optimizer.zero_grad()
        loss, loss_each = self.compute_loss(inputs, data, model, it)

        loss.backward()
        grad_norm = inputs.grad.view(-1).norm(2).item() if inputs.grad is not None else 0.0
        self.optimizer.step()
        
        return loss.item(), loss_each, grad_norm
    
    def compute_loss(self, inputs:torch.Tensor, data:Dict[str,torch.Tensor], model:Optional[torch.nn.Module], it:int=0):
        '''  Compute the loss.
        Args:
            data (dict)              : data dictionary
            inputs (torch.tensor)    : input point clouds
            model (nn.Module or None): a neural network or None
            it (int)                 : the number of iterations
        '''

        device = self.device
        res = self.cfg['model']['grid_res']
        
        # source oriented point clouds to PSR grid
        psr_grid, points, normals = self.pcl2psr(inputs)
        
        # build mesh
        v, f, n = self.psr2mesh(psr_grid)
        
        # the output is in the range of [0, 1), we make it to the real range [0, 1]. 
        # This is a hack for our DPSR solver
        v = v * res / (res-1) 

        points = points * 2. - 1.
        v = v * 2. - 1. # within the range of (-1, 1)

        loss = 0
        loss_each = {}
        # compute loss
        if self.data_type == 'point':
            if self.cfg['train']['w_chamfer'] > 0:
                loss_ = self.cfg['train']['w_chamfer'] * self.compute_3d_loss(v,f, data)
                loss_each['chamfer'] = loss_
                loss += loss_
        elif self.data_type == 'img': 
            loss, loss_each = self.compute_2d_loss(inputs, data, model)
    
        return loss, loss_each
    
    def compute_3d_loss(self, v:torch.Tensor, f:torch.Tensor,data:Dict[str,torch.Tensor])->torch.Tensor:
        '''  Compute the loss for point clouds.
        Args:
            v (torch.tensor)         : mesh vertices #(1, V, 3)
            f (torch.tensor)         : mesh faces #(1, F, 3)
            data (dict)              : data dictionary
        '''

        pts_gt = data['target_points']
        idx = np.random.randint(pts_gt.shape[1], size=self.cfg['train']['n_sup_point'])
        if self.cfg['train']['subsample_vertex']:
            #chamfer distance only on random sampled vertices
            idx = np.random.randint(v.shape[1], size=self.cfg['train']['n_sup_point'])
            v_mesh = v[:, idx]
        else:
            v_mesh = v

        x_0_weight = torch.tensor([self.cfg['train']['w_chamfer_x_0']]).view(1,1).expand(-1,pts_gt.shape[1]).to(self.device)
        v_mesh_weight = torch.tensor([self.cfg['train']['w_chamfer_v_mesh']]).view(1,1).expand(-1,v_mesh.shape[1]).to(self.device)
        
        occ = check_sign(v,f[0].to(torch.int64),pts_gt)
        x_0_weight[occ] *= self.cfg['train']['w_inside']
        x_0_weight[~occ] *= self.cfg['train']['w_outside']
        
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(v[0].detach().cpu().numpy())
        # mesh.triangles = o3d.utility.Vector3iVector(f[0].detach().cpu().numpy())
        
        # tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # scene = o3d.t.geometry.RaycastingScene()
        # _ = scene.add_triangles(tmesh)

        # occ = scene.compute_occupancy(o3d.core.Tensor(pts_gt[0].detach().cpu().numpy(), dtype=o3d.core.Dtype.Float32))
        # x_0_weight[:,occ.numpy() > 0] *= self.cfg['train']['w_inside']
        # x_0_weight[:,occ.numpy() <= 0] *= self.cfg['train']['w_outside']
        
        loss, _ = chamfer_distance_surface(
            x_0 = pts_gt,
            v_mesh = v_mesh,
            x_0_weight=x_0_weight,
            v_mesh_weight=v_mesh_weight,
            batch_reduction="mean",
            point_reduction="mean",
            norm=2,
            abs_cosine=True,
        )

        return loss
    
    def compute_2d_loss(self, inputs, data, model):
        '''  Compute the 2D losses.
        Args:
            inputs (torch.tensor)    : input source point clouds
            data (dict)              : data dictionary
            model (nn.Module or None): neural network or None
        '''
        
        losses = {"color": 
                    {"weight": self.cfg['train']['l_weight']['rgb'], 
                     "values": []
                    },
                  "silhouette": 
                    {"weight": self.cfg['train']['l_weight']['mask'], 
                     "values": []},
                }
        loss_all = {k: torch.tensor(0.0, device=self.device) for k in losses}
            
        # forward pass
        out = model(inputs, data)

        if out['rgb'] is not None:
            rgb_gt = out['rgb_gt'].reshape(self.cfg['data']['n_views_per_iter'], 
                                                -1, 3)[out['vis_mask']]
            loss_all["color"] += torch.nn.L1Loss(reduction='sum')(rgb_gt,
                                        out['rgb']) / out['rgb'].shape[0]

        if out['mask'] is not None:
            loss_all["silhouette"] += ((out['mask'] - out['mask_gt']) ** 2).mean()  
            
        # weighted sum of the losses
        loss = torch.tensor(0.0, device=self.device)
        for k, l in loss_all.items():
            loss += l * losses[k]["weight"]
            losses[k]["values"].append(l)

        return loss, loss_all
    # endregion
    
    # region Utils
    def pcl2psr(self, inputs:torch.Tensor):
        '''  Convert an oriented point cloud to PSR indicator grid
        Args:
            inputs (torch.tensor): input oriented point clouds
        '''

        points, normals = inputs[...,:3], inputs[...,3:]
        if self.cfg['model']['apply_sigmoid']:
            points = torch.sigmoid(points)
        if self.cfg['model']['normal_normalize']:
            normals = normals / normals.norm(dim=-1, keepdim=True)

        # DPSR to get grid
        psr_grid = self.dpsr(points, normals).unsqueeze(1)
        psr_grid = torch.tanh(psr_grid)

        return psr_grid, points, normals
    
    def point_resampling(self, inputs):
        '''  Resample points
        Args:
            inputs (torch.tensor): oriented point clouds
        '''
    
        psr_grid, points, normals = self.pcl2psr(inputs)
        
        # shortcuts
        n_grow = self.cfg['train']['n_grow_points']

        # [hack] for points resampled from the mesh from marching cubes, 
        # we need to divide by s instead of (s-1), and the scale is correct.
        verts, faces, _ = mc_from_psr(psr_grid, real_scale=False, zero_level=0)

        # find the largest component
        pts_mesh, faces_mesh = verts_on_largest_mesh(verts, faces)
    
        # sample vertices only from the largest component, not from fragments
        mesh = trimesh.Trimesh(vertices=pts_mesh, faces=faces_mesh)
        pi, face_idx = mesh.sample(n_grow+points.shape[1], return_index=True)
        normals_i = mesh.face_normals[face_idx].astype('float32')
        pts_mesh = torch.tensor(pi.astype('float32')).to(self.device)[None]
        n_mesh = torch.tensor(normals_i).to(self.device)[None]

        points, normals = pts_mesh, n_mesh
        # print('{} total points are resampled'.format(points.shape[1]))
    
        # update inputs
        points = torch.log(points / (1 - points)) # inverse sigmoid
        inputs = torch.cat([points, normals], dim=-1)
        inputs.requires_grad = True  

        return inputs
    
    def save_mesh_pointclouds(self, inputs, epoch, log_dir:str, center=None, scale=None):
        '''  Save meshes and point clouds.
        Args:
            inputs (torch.tensor)       : source point clouds
            epoch (int)                 : the number of iterations
            log_dir (path)              : the path to store the log
            center (numpy.array)        : center of the shape
            scale (numpy.array)         : scale of the shape
        '''
        exp_pcl = self.cfg['train']['exp_pcl']
        exp_mesh = self.cfg['train']['exp_mesh']
        
        psr_grid, points, normals = self.pcl2psr(inputs)
        
        if exp_pcl:
            dir_pcl = os.path.join(log_dir,'pcd')
            os.makedirs(dir_pcl,exist_ok=True)
            p = points.squeeze(0).detach().cpu().numpy()
            p = p * 2 - 1
            n = normals.squeeze(0).detach().cpu().numpy()
            if scale is not None:
                p *= scale
            if center is not None:
                p += center
            export_pointcloud(os.path.join(dir_pcl, 'pcd_{:04d}.ply'.format(epoch)), p, n)
        if exp_mesh:
            dir_mesh = os.path.join(log_dir,'mesh')
            os.makedirs(dir_mesh,exist_ok=True)
            with torch.no_grad():
                v, f, _ = mc_from_psr(psr_grid,
                        zero_level=self.cfg['data']['zero_level'], real_scale=True)
                v = v * 2 - 1
                if scale is not None:
                    v *= scale
                if center is not None:
                    v += center
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v)
            mesh.triangles = o3d.utility.Vector3iVector(f)
            outdir_mesh = os.path.join(dir_mesh, 'mesh_{:04d}.ply'.format(epoch))
            o3d.io.write_triangle_mesh(outdir_mesh, mesh)
            
    def export_mesh(self, inputs, center=None, scale=None):
        psr_grid, points, normals = self.pcl2psr(inputs)
        with torch.no_grad():
            v, f, _ = mc_from_psr(psr_grid,zero_level=self.cfg['data']['zero_level'], real_scale=True)
            v = v * 2 - 1
            if scale is not None:
                v *= scale
            if center is not None:
                v += center
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(f)
        return mesh
        
    # endregion