import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from typing import TYPE_CHECKING, Dict, List, Optional, Callable, Literal,Union
import matplotlib.pyplot as plt
import logging

from .base import Base
from .utils import calibrate_translate_pts,rotate_y,find_mid_lowest_pt,visualize_point_cloud
from softzoo.utils.general_utils import extract_part_pca_inner,row_permutation,extract_part_pca
from softzoo.utils.computation_utils import directions_to_spherical
from softzoo.utils.visualization_utils import get_arrow

if TYPE_CHECKING:
    from softzoo.envs.base_env import BaseEnv

class GeneratedPointEPCD(Base):
    def __init__(
        self,lr,
        env:'BaseEnv', n_voxels:int=20, sigma:float = 7e-4,
        passive_geometry_mul:int=1,passive_softness_mul:int=1,
        device:str='cpu',
        bounding_box:Optional[Dict[Literal['max','mean','min'],List[float]]] = None,
        **kwargs
    ):
        super(GeneratedPointEPCD, self).__init__(env)
        self.passive_geometry_mul = passive_geometry_mul
        self.passive_softness_mul = passive_softness_mul
        self.bounding_box = bounding_box
        
        self.original_coords = self.env.design_space.get_x(s=0).float()
        if isinstance(self.original_coords, np.ndarray):
            self.original_coords = torch.from_numpy(self.original_coords).float()
            
        self.sigma = sigma
        
        self.device = torch.device(device)
        self.to(self.device)
        self.original_coords = self.original_coords.to(self.device)
        
    def calculate_base_location_parallel_gripper(self):
        base_pcd = o3d.io.read_point_cloud('diff_conditioning/simulation_env/asset/fixed_base_big.pcd')
        base_points = torch.from_numpy(np.asarray(base_pcd.points)).to(self.device)
        calibrated_base_pts = calibrate_translate_pts(
            base_points, 
            mean = torch.tensor([0.,0.,0.]).to(self.device), 
            scale=torch.tensor([1.,1.,1.]).to(self.device)
        )
        base_labels = [0 for _ in range(base_points.shape[0])]
        return calibrated_base_pts,np.array(base_labels)
        
    def calculate_base_location(self,gripper_pos_np:np.ndarray):
        # Label 0 is reserved for the base, where no velocity is allowed to propagate to hold the gripper up.
        # The rest of the labels comes in pair (2n+1,2n+2) denotes the muscle pair n.
        
        base_pcd = o3d.io.read_point_cloud('diff_conditioning/simulation_env/asset/fixed_base_big.pcd')
        base_points = np.asarray(base_pcd.points)
        
        base_coords_min, base_coords_max = base_points.min(0),base_points.max(0)
        gripper_points_min, gripper_points_max = gripper_pos_np.min(0),gripper_pos_np.max(0)
        base_extent = base_coords_max - base_coords_min
        gripper_extent = gripper_points_max - gripper_points_min
        gripper_base_scale = gripper_extent/base_extent
        
        lowest_pt = torch.from_numpy(find_mid_lowest_pt(gripper_pos_np,0.5)).to(self.device)
    
        calibrated_base_pts = calibrate_translate_pts(
            torch.from_numpy(base_points).to(self.device), 
            mean = lowest_pt, 
            scale=torch.tensor([
                0.1*gripper_base_scale[0],
                0.02*gripper_base_scale[1],
                0.1*gripper_base_scale[2]
            ]).to(self.device)
        )
    
        base_labels = [0]*base_points.shape[0]
        
        return calibrated_base_pts,np.array(base_labels)
    
    # region Muscle Label Generation
    def get_muscle_label(
        self,
        gripper_pos_np:np.ndarray,
        normalizing:bool=False,pca_components_index:List[int] = [],
        muscle_count:int = 12, 
        target_center:List[float] = [0.,1.,0.],
        feature_selector:Callable[[np.ndarray],np.ndarray] = lambda x:x[:,[0,2]]
    ):
        assert muscle_count%2==0, "Muscle count must be an even number"
        
        points = gripper_pos_np
        points_std = StandardScaler().fit_transform(points)
        coarse_labels = self._generate_coarse_muscle(points_std,normalizing,pca_components_index,muscle_count,feature_selector=feature_selector)
        # visualize_point_cloud(torch.from_numpy(points_std),coarse_labels)
        
        max_label = coarse_labels.max()
        target_center_np = np.array(target_center)
        additional_clusters = []
        for c in range(max_label + 1):
            cluster_idx = np.where(coarse_labels == c)[0]
            cluster_points = points_std[cluster_idx]
            if cluster_points.shape[0] == 0:
                logging.warning(f"Coarse muscle clusters {c} skipped due to no members available.")
                continue
            
            additional_clusters += self._split_muscle_layers(cluster_points,target_center_np,split_mode='fixed')
        additional_clusters = np.array(additional_clusters)  # (m,3)
        labels = pairwise_distances_argmin(points_std, additional_clusters) 
        max_label = labels.max()
        assert max_label < 60, "Too many clusters, please reduce the number of clusters below 60."
        labels[labels<0] = 0
        labels += 1
        return labels
    
    def _generate_coarse_muscle(
        self,points_std: np.ndarray,
        normalizing:bool=False,pca_components_index:List[int] = [],
        muscle_count:int = 16,
        feature_selector:Callable[[np.ndarray],np.ndarray] = lambda x:x
    ):  
        points_std = feature_selector(points_std)
        if normalizing:
            n_components = min(3,points_std.shape[1])
            pca = PCA(n_components=n_components)
            feats = pca.fit_transform(points_std)
        else:
            feats = points_std
        
        
        assert len(pca_components_index) <=3, "Only a maximum of 3 comp index is allowed."
        if len(pca_components_index) > 0:
            feats_centered = feats - feats.mean(0)[None,:] # already centered by standardizer but just in case
            hyperplane_sdf = []
            for comp_i in pca_components_index:
                vec = pca.components_[comp_i, :]
                hp_sdf = (feats_centered * vec[None,:]).sum(1)
                hyperplane_sdf.append(hp_sdf)
            hyperplane_sdf = np.stack(hyperplane_sdf, axis=-1)
            feats = np.concatenate([feats, hyperplane_sdf], axis=1)
        
        n_clusters = muscle_count//2
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42,)
        kmeans.fit(feats)
        labels = kmeans.labels_
        return labels
    
    def _split_muscle_layers(self, cluster_points:np.ndarray, target_center_np:np.ndarray,split_mode:Literal['pca','target_pt','fixed']='fixed'):
        # PCA on cluster points
        cluster_center = cluster_points.mean(axis=0)
        if split_mode == 'pca':
            # TODO: Fix problem here sometimes num samples < 3
            if min(*cluster_points.shape,3)<3: logging.warning("Warning!!!: Numer of points in cluster is small <3.")
            pca_local = PCA(n_components=min(*cluster_points.shape,3))
            pca_local.fit(cluster_points)

            # Use 3rd component as local "vertical" axis
            current_vector_up = target_center_np - cluster_center
            norm_current_vector_up = current_vector_up/np.linalg.norm(current_vector_up)
            
            dot_products = [np.dot(comp, norm_current_vector_up) for comp in pca_local.components_]
            best_idx = np.argmax(np.abs(dot_products))
            local_up = pca_local.components_[best_idx]

            # Flip if necessary to ensure it's pointing toward the grasp center
            if dot_products[best_idx] < 0:
                local_up = -local_up
        elif split_mode =='target_pt':
            diff_target_center = target_center_np - cluster_center
            local_up = diff_target_center/np.linalg.norm(diff_target_center)
        elif split_mode == 'fixed':
            local_up = target_center_np/np.linalg.norm(target_center_np)
                        
        proj = cluster_points @ local_up
        proj_min, proj_max = proj.min(), proj.max()
        p1 = proj_min + (proj_max - proj_min) / 3
        p2 = proj_min + 2 * (proj_max - proj_min) / 3

        # Reconstruct back into 3D coordinates for clustering centers
        center1 = cluster_center + local_up * (p1 - proj.mean())
        center2 = cluster_center + local_up * (p2 - proj.mean())
        return [center1,center2]
    # endregion

    @property
    def has_actuator_direction(self):
        return True
    def save_actuator_direction(self,design:Dict[str,torch.Tensor],save_path:str):
        np.save(save_path,design['actuator_direction'].detach().cpu().numpy())
    
    def reset(self):
        self.out_cache:Dict[str,Optional[torch.Tensor]] = dict(
            geometry=None, 
            softness=None, 
            actuator=None,
            actuator_direction=None,
            is_passive_fixed = None
        )
    
    def forward(self,input:torch.Tensor,num_fingers:int):
        inp = input.to(self.device)
        
        self.create_representation_from_tensor(inp,num_fingers)
        
        active_geometry = self.occupancy
        passive_geometry = self.occupancy * self.passive_geometry_mul # NOTE: the same as active
        geometry = torch.where(self.is_passive==1., passive_geometry, active_geometry)
        
        active_softness = torch.ones_like(self.occupancy)
        passive_softness = torch.ones_like(self.occupancy) * self.passive_softness_mul # NOTE: the same as active
        softness = torch.where(self.is_passive==1., passive_softness, active_softness)
        
        self.out_cache['geometry'] = geometry
        self.out_cache['softness'] = softness
        self.out_cache['actuator'] = self.actuator
        self.out_cache['actuator_direction'] = self.actuator_directions
        self.out_cache['is_passive_fixed'] = self.is_passive.to(torch.float32)

        # NOTE: must use clone here otherwise tensor may be modified in-place in sim
        design = {k: v.clone() for k, v in self.out_cache.items()}
        return design
    
    def create_representation_from_tensor(self,gripper_pos_tensor:torch.Tensor,num_fingers:int):
        # region Preprocess and Attach Base
        gripper_pos_np = gripper_pos_tensor.detach().cpu().numpy() # [N,C]
        gripper_labels = self.get_muscle_label(gripper_pos_np,feature_selector=lambda x:x[:,[1]],target_center=[1.,0.,0.],muscle_count=4)
        max_label = gripper_labels.max()
        max_gripper_y = gripper_pos_tensor.max(0).values[1]
        
        base_pos_tensor, base_labels = self.calculate_base_location_parallel_gripper()
        max_base_y = base_pos_tensor.max(0).values[1]
        
        # region Assigning Fingers to Base
        angles = torch.arange(num_fingers, device=self.device) * (2 * torch.pi / num_fingers)
        x_axis_pos = torch.cos(torch.pi - angles) * 0.5
        z_axis_pos = torch.sin(torch.pi - angles) * 0.5
        y_axis_pos = torch.zeros_like(x_axis_pos) + max_base_y - max_gripper_y
        
        fingers_pos = torch.stack([x_axis_pos, y_axis_pos, z_axis_pos], dim=1)  # [N,3]
        
        complete_pos = [base_pos_tensor]
        complete_lbls = [base_labels]
        
        for i in range(num_fingers):
            finger_i = rotate_y(gripper_pos_tensor.clone(),angles[i])
            calibrated_finger_i = calibrate_translate_pts(finger_i,mean=fingers_pos[i])
            complete_pos.append(calibrated_finger_i)
            complete_lbls.append(gripper_labels.copy() + max_label)
        # endregion
        
        complete_pos_tensor = torch.concat(complete_pos,dim=0).float()
        complete_labels_np = np.concatenate(complete_lbls,axis=0)
    
        # visualize_point_cloud(complete_pos_tensor,complete_labels)
        unique_lbls = np.unique(complete_labels_np)
        if not self.env.sim.solver.n_actuators == unique_lbls.shape[0]:
            logging.warning(f"Warning!!!!: \n The number of actuators {self.env.sim.solver.n_actuators} must be equal to the number of generated labels {unique_lbls.shape[0]}. Probllem in configuration files")
        # endregion
        
        # region Calculate Actuator Direction
        all_part_pca_components, all_part_pca_singular_values, all_part_pc = extract_part_pca_inner(
            complete_pos_tensor.detach().cpu().numpy(),
            complete_labels_np,
            unique_lbls=range(self.env.sim.solver.n_actuators)
        )        
        actuator_directions = [] # TODO: make dict
        for k, part_pca_component in all_part_pca_components.items():
            # Taking the first PCA component as the direction for actuators
            actuator_directions.append(part_pca_component[0])
        actuator_directions = np.array(actuator_directions)
        actuator_directions = directions_to_spherical(actuator_directions)
        self.actuator_directions = nn.Parameter(torch.from_numpy(actuator_directions))
        # endregion
        
        # region Calculate Occupancy (!!!!! Ensure Differentiability)
        coords_min, coords_mean, coords_max = self.original_coords.min(0).values, self.original_coords.mean(0), self.original_coords.max(0).values
        actual_min,actual_mean,actual_max = complete_pos_tensor.min(0).values,complete_pos_tensor.mean(0),complete_pos_tensor.max(0).values 
        
        points_min = actual_min if self.bounding_box is None or 'min' not in self.bounding_box else torch.min(torch.tensor(self.bounding_box['min']).to(self.device),actual_min)
        points_mean =  actual_mean if self.bounding_box is None or 'mean' not in self.bounding_box else torch.tensor(self.bounding_box['mean']).to(self.device)
        points_max = actual_max if self.bounding_box is None or 'max' not in self.bounding_box else torch.max(torch.tensor(self.bounding_box['max']).to(self.device),actual_max)

        def calibrate_points(_pts:torch.Tensor, y_offset=0.):
            _pts_calibrated = _pts - points_mean # center
            _pts_calibrated = _pts_calibrated / torch.max(points_max - points_min) * torch.max(coords_max - coords_min)*0.95 # rescale
            _pts_calibrated = _pts_calibrated + coords_mean # recenter
            _pts_calibrated = _pts_calibrated + torch.clip(coords_min - _pts_calibrated.min(0).values, min=0, max=torch.inf) # make sure within min-bound
            _pts_calibrated = _pts_calibrated - torch.clip(_pts_calibrated.max(0).values - coords_max, min=0, max=torch.inf) # make sure within max-bound
            _pts_calibrated[:,1] = _pts_calibrated[:,1] + y_offset # align lower bound in y-axis
            return _pts_calibrated
        complete_pos_tensor_calibrated = calibrate_points(complete_pos_tensor)
        y_offset = coords_max[1] - complete_pos_tensor_calibrated.max(0).values[1] # To make the gripper at the top of the design space
        complete_pos_tensor_calibrated = calibrate_points(complete_pos_tensor,y_offset=y_offset)
        
        pairwise_dist = torch.cdist(self.original_coords,complete_pos_tensor_calibrated) # , sim_coords(n), generated_coords(m)
        p_ji = torch.exp(-(pairwise_dist**2) / (2 * self.sigma**2))  # (n, m) # Smooth per-point contribution (Gaussian kernel)
        self.occupancy = 1 - torch.prod(1 - p_ji, dim=1)  # (n,) # Soft OR across points → final occupancy per voxel
        # endregion

        # region Cluster Sim Coords
        passive_lbl = 0
        self.actuator = torch.zeros((self.env.sim.solver.n_actuators, self.original_coords.shape[0])) #(n_act,n)
        self.is_passive = torch.zeros((self.original_coords.shape[0])).bool() #(n)
        
        coords_lbls_prob = torch.zeros((self.original_coords.shape[0],self.env.sim.solver.n_actuators))
        for lbl in unique_lbls:
            coords_cur_lbls_prob = p_ji[:,complete_labels_np==lbl]
            occupancy_cluster = (1 - torch.prod(1-coords_cur_lbls_prob,dim=1)) #*(1. if lbl!=passive_lbl else 1e4)
            coords_lbls_prob[:,lbl] = occupancy_cluster
        coords_cluster = coords_lbls_prob.argmax(dim=1)
        for lbl in unique_lbls:
            if lbl==passive_lbl: self.is_passive[coords_cluster==lbl] = 1.
            else: self.actuator[lbl,coords_cluster==lbl] = 1.
        
        # print(self.is_passive[(self.is_passive + self.occupancy)>1.].sum())
        # endregion
        

