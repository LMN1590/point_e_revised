import taichi as ti
import open3d as o3d
import numpy as np
import torch
import geomloss

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE
from softzoo.utils.general_utils import load_points_from_mesh

import os
from typing import TYPE_CHECKING,List,Optional
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv
    
@ti.data_oriented
class AnimatedEMDLoss(Loss):
    '''
    Guide the deformable robot to match certain deformation 
    '''
    
    def __init__(
        self,
        parent:'LossSet', env:'BaseEnv',
        mesh_dir:str, substep_freq:int, mesh_num_points:int,
        final_target_idx:Optional[int]=None, 
        recenter_mesh_target:bool=True, no_reset_offset:bool=False,no_normalize_by_n_particles:bool=False
    ):
        super().__init__(parent,env)
        
        assert env.sim.device.name in ['TorchCPU', 'TorchGPU']
        
        self.substep_freq = substep_freq
        self.final_target_idx = final_target_idx
        self.recenter_mesh_target = recenter_mesh_target
        self.no_reset_offset = no_reset_offset
        self.no_normalize_by_n_particles = no_normalize_by_n_particles
        
        self.points = []
        for mesh_fname in sorted(os.listdir(mesh_dir)):
            file_ext = os.path.splitext(mesh_fname)[-1]
            mesh_fpath = os.path.abspath(os.path.join(mesh_dir, mesh_fname))
            if file_ext == '.obj':
                mesh_scale = self.env.design_space.cfg.base_shape.scale
                mesh_offset = self.env.design_space.cfg.base_shape.offset
                points_i = load_points_from_mesh(mesh_fpath, scale=mesh_scale, offset=mesh_offset, num_points=mesh_num_points)
            elif file_ext == '.pcd':
                pcd = o3d.io.read_point_cloud(mesh_fpath)
                points_i = np.asarray(pcd.points)
            else:
                continue
            
            points_i = env.sim.device.tensor(points_i)
            self.points.append(points_i)
        
        self.loss_F = geomloss.SamplesLoss(
            loss='sinkhorn',
            p=2,
            blur=0.01,
            debias=False,
            potentials=False
        )
        self.data = dict(
            loss = ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps),needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True)
        )
    
    def reset(self):
        super().reset()
        self._compute_n_robot_particles()
        x0 = self.env.design_space.get_x(0)
        self.x0_mean = x0.mean(0)
        self._grad = dict()
        
    def compute_final_step_loss(self, s):
        if self.final_target_idx is not None:
            self._compute_step_loss(s, self.final_target_idx)

    def compute_final_step_grad(self, s):
        if self.final_target_idx is not None:
            self._compute_step_grad(s)

    def compute_per_step_loss(self, s):
        if (s % self.substep_freq == 0) and (self.final_target_idx is None):
            v_i = (s // self.substep_freq) % len(self.points)
            self._compute_step_loss(s, v_i)
    
    def compute_per_step_grad(self, s):
        if (s % self.substep_freq == 0) and (self.final_target_idx is None):
            self._compute_step_grad(s)
            
    def _compute_step_loss(self, s, v_i):
        x = self.env.design_space.get_x(s) # TODO: cannot handle geometry codesign for now
        x.requires_grad = True

        x_target = self.points[v_i]
        if self.recenter_mesh_target:
            offset = x.mean(0)
        else:
            offset = self.x0_mean
        if not self.no_reset_offset:
            x_target = x_target - x_target.mean(0) + offset # NOTE: do we need rotation?!

        L = self.loss_F(x, x_target)
        if not self.no_normalize_by_n_particles:
            L = L / self.data['n_robot_particles'][None]
        self.data['loss'][s] = L.item()

        if False:
            import numpy as np
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x.data.cpu().numpy())
            pcd.paint_uniform_color((0., 1., 0.))
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(x_target.data.cpu().numpy())
            pcd_target.paint_uniform_color((1., 0., 0.))
            # o3d.visualization.draw_geometries([pcd, pcd_target])

            pcd_merged = o3d.geometry.PointCloud()
            pcd_merged.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.points), np.asarray(pcd_target.points)]))
            pcd_merged.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.colors), np.asarray(pcd_target.colors)]))
            o3d.io.write_point_cloud('./local/tmp.pcd', pcd_merged)
            import pdb; pdb.set_trace()

        g_x, = torch.autograd.grad(L, [x])
        self._grad[s] = g_x

    def _compute_step_grad(self, s):
        # NOTE: assume shape not changing
        s_local = self.get_s_local(s)
        mask = self.env.design_space.get_particle_mask()
        n_active_particles = mask.sum().item()
        mask = mask[mask>0]
        mask = mask[:, None]
        grad = self.env.sim.device.create_f_tensor((n_active_particles, self.env.sim.solver.dim))
        grad = self._grad[s] * mask + grad * (1 - mask) # TODO: setting grad like this can be slow with many non-robot particles
        self.add_design_x_grad(s_local, grad)

    @ti.kernel
    def add_design_x_grad(self, s: I_DTYPE, grad: ti.types.ndarray()):
        # for p in range(self.env.sim.solver.n_particles[None]):
        for p in range(self.env.design_space.p_start, self.env.design_space.p_end):
            p_ext = p - self.env.design_space.p_start
            id = self.env.sim.solver.particle_ids[p]
            if self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                for d in ti.static(range(self.env.sim.solver.dim)):
                    self.env.sim.solver.x.grad[s, p][d] += grad[p_ext, d]