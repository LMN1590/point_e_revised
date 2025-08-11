import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import TYPE_CHECKING, Dict

from .base import Base
from softzoo.utils.general_utils import extract_part_pca,row_permutation
from softzoo.utils.computation_utils import directions_to_spherical

if TYPE_CHECKING:
    from softzoo.envs.base_env import BaseEnv

class AnnotatedPCD(Base):
    def __init__(
        self,lr,
        env:'BaseEnv',pcd_path:str,n_voxels:int=20,
        passive_geometry_mul:int=1,passive_softness_mul:int=1,
        device:str='cpu',
        **kwargs
    ):
        super(AnnotatedPCD, self).__init__(env)

        self.passive_geometry_mul = passive_geometry_mul
        self.passive_softness_mul = passive_softness_mul

        # Load pcd
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        
        # region Get Muscle
        # Get actuator group and actuator direction
        all_part_pca_components, all_part_pca_singular_values, all_part_pc, all_part_colors = extract_part_pca(pcd, return_part_colors=True)
        actuator_directions = [] # TODO: make dict
        for k, part_pca_component in all_part_pca_components.items():
            # Taking the first PCA component as the direction for actuators
            actuator_directions.append(part_pca_component[0])
        actuator_directions = np.array(actuator_directions)
        actuator_directions = directions_to_spherical(actuator_directions)
        self.actuator_directions = nn.Parameter(torch.from_numpy(actuator_directions))
        # endregion
        
        # region Load PCD
        # Get voxel grid based on part point cloud
        coords = self.env.design_space.get_x(s=0).float()
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
            
        coords_min, coords_mean, coords_max = coords.min(0), coords.mean(0), coords.max(0)
        points_min, points_mean, points_max = points.min(0), points.mean(0), points.max(0)
        def calibrate_points(_pts, y_offset=0.):
            _pts_calibrated = _pts - points_mean # center
            _pts_calibrated = _pts_calibrated / max(points_max - points_min) * max(coords_max - coords_min) # rescale
            _pts_calibrated = _pts_calibrated + coords_mean # recenter
            _pts_calibrated = _pts_calibrated + np.clip(coords_min - _pts_calibrated.min(0), a_min=0, a_max=np.inf) # make sure within min-bound
            _pts_calibrated = _pts_calibrated - np.clip(_pts_calibrated.max(0) - coords_max, a_min=0, a_max=np.inf) # make sure within max-bound
            _pts_calibrated[:,1] = _pts_calibrated[:,1] + y_offset # align lower bound in y-axis
            return _pts_calibrated
        points_calibrated = calibrate_points(points)
        y_offset = coords_min[1] - points_calibrated.min(0)[1] # The supposedly distance between the lowest point of the point cloud and the bouding box, to ensure it is at the bottom.
        
        voxel_size = max(points_calibrated.max(0) - points_calibrated.min(0)) / n_voxels
        
        all_part_voxel_grid = dict()
        for k, part_pc in all_part_pc.items():
            part_pc_calibrated = calibrate_points(part_pc, y_offset=y_offset)

            part_pcd = o3d.geometry.PointCloud()
            part_pcd.points = o3d.utility.Vector3dVector(part_pc_calibrated)
            all_part_voxel_grid[k] = o3d.geometry.VoxelGrid.create_from_point_cloud(part_pcd, voxel_size=voxel_size)
        # endregion
        
        # region Assign Points to Muscle Clusters
        # Get occupancy, actuator placement based on voxel grid
        passive_color = np.array([0, 0, 0])
        passive_k = np.where(np.all(all_part_colors == passive_color, axis=1))[0][0]
        is_passive = np.zeros((coords.shape[0]), dtype=bool)
        
        occupancy = np.zeros((coords.shape[0]), dtype=bool)
        actuator = np.zeros((self.env.sim.solver.n_actuators, coords.shape[0]), dtype=float)
        for k, part_voxel_grid in all_part_voxel_grid.items():
            is_part = part_voxel_grid.check_if_included(o3d.utility.Vector3dVector(coords))
            occupancy = np.logical_or(occupancy, is_part)
            if k == passive_k:
                is_passive = np.logical_or(is_passive, is_part)
            else:
                actuator[k, is_part] = 1.
        
        for k in all_part_voxel_grid.keys(): # make sure every particle belongs to a single actuator
            if k > 0:
                has_actuator = np.any(actuator[:k, :] > 0, axis=0)
                actuator[k, has_actuator] = 0    
        assert actuator.sum(0)[occupancy].min() in [0, 1] and actuator.sum(0)[occupancy].max() in [0, 1]

        self.original_coords = nn.Parameter(torch.from_numpy(coords),requires_grad=False)
        self.is_passive = torch.from_numpy(is_passive)

        self.occupancy = nn.Parameter(torch.from_numpy(occupancy).float())
        self.actuator = nn.Parameter(torch.from_numpy(actuator))
        
        self.optim = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device(device)
        self.to(self.device)
        # endregion
        
    def reset(self):
        self.out_cache:Dict[str,torch.Tensor] = dict(
            geometry=None, 
            softness=None, 
            actuator=None,
            actuator_direction=None,
            is_passive_fixed = None
        )

    def forward(self, inp=None):
        # actuator_directions_matrix = []
        # for i in range(self.env.sim.solver.n_actuators):
        #     if i < self.actuator_directions.shape[0]:
        #         AAt = self.actuator_directions[i].reshape(-1, 1) @ self.actuator_directions[i].reshape(-1, 1).T
        #     else:
        #         AAt = torch.zeros((self.env.sim.solver.dim, self.env.sim.solver.dim))
        #     actuator_directions_matrix.append(AAt)
        # actuator_directions_matrix = torch.stack(actuator_directions_matrix, dim=0)

        active_geometry = self.occupancy
        passive_geometry = self.occupancy * self.passive_geometry_mul # NOTE: the same as active
        geometry = torch.where(self.is_passive, passive_geometry, active_geometry)

        active_softness = torch.ones_like(self.occupancy)
        passive_softness = torch.ones_like(self.occupancy) * self.passive_softness_mul # NOTE: the same as active
        softness = torch.where(self.is_passive, passive_softness, active_softness)
        
        self.out_cache['geometry'] = geometry
        self.out_cache['softness'] = softness
        self.out_cache['actuator'] = self.actuator
        self.out_cache['actuator_direction'] = self.actuator_directions
        self.out_cache['is_passive_fixed'] = self.is_passive.to(torch.float32)

        # NOTE: must use clone here otherwise tensor may be modified in-place in sim
        design = {k: v.clone() for k, v in self.out_cache.items()}
        return design
    
    def visualization_dev(self, coords, points_calibrated, all_part_pc, all_part_voxel_grid, voxel_size):
        import matplotlib.pyplot as plt

        coords_pcd = o3d.geometry.PointCloud()
        coords_pcd.points = o3d.utility.Vector3dVector(coords)

        points_calibrated_pcd = o3d.geometry.PointCloud()
        points_calibrated_pcd.points = o3d.utility.Vector3dVector(points_calibrated)
        points_calibrated_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(points_calibrated_pcd, voxel_size=voxel_size)
        o3d.visualization.draw_geometries([coords_pcd, points_calibrated_voxel_grid])
        # o3d.visualization.draw_geometries([coords_pcd, points_calibrated_pcd])
        # o3d.visualization.draw_geometries([points_calibrated_pcd])
        # o3d.visualization.draw_geometries_with_vertex_selection([points_calibrated_pcd])

        # is_occupied = points_calibrated_voxel_grid.check_if_included(o3d.utility.Vector3dVector(coords))

        all_part_pcd_vis = []
        labels = np.array(list(all_part_pc.keys()))
        max_label = labels.max()
        vis_colors = plt.get_cmap('tab20')(labels / (max_label if max_label > 0 else 1))
        for k, part_voxel_grid in all_part_voxel_grid.items():
            is_part = part_voxel_grid.check_if_included(o3d.utility.Vector3dVector(coords))
            # is_part = np.logical_and(is_occupied, is_part)

            part_pcd_vis = o3d.geometry.PointCloud()
            part_pcd_vis.points = o3d.utility.Vector3dVector(coords[is_part])
            part_pcd_vis.paint_uniform_color(vis_colors[k][:3])
            all_part_pcd_vis.append(part_pcd_vis)
        
        coords_pcd = o3d.geometry.PointCloud()
        coords_pcd.points = o3d.utility.Vector3dVector(coords)
        o3d.visualization.draw_geometries(all_part_pcd_vis)
        # o3d.visualization.draw_geometries(list(all_part_voxel_grid.values())) 

    @property
    def has_actuator_direction(self):
        return True
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        orig_coords = checkpoint['model_state_dict']['original_coords']
        new_coords = self.original_coords
        
        perm_idx = row_permutation(orig_coords,new_coords)
        self.occupancy.data.copy_(checkpoint['model_state_dict']['occupancy'][perm_idx].float())
        self.actuator.data.copy_(checkpoint['model_state_dict']['actuator'].T[perm_idx].T.float())
        self.actuator_directions.data.copy_(checkpoint['model_state_dict']['actuator_directions'].float())
        
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])