import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from typing import TYPE_CHECKING, Dict, List, Optional
import matplotlib.pyplot as plt

from .base import Base
from softzoo.utils.general_utils import extract_part_pca_inner,row_permutation
from softzoo.utils.computation_utils import directions_to_spherical

if TYPE_CHECKING:
    from softzoo.envs.base_env import BaseEnv

class GeneratedPointEPCD(Base):
    def __init__(
        self,lr,
        env:'BaseEnv',
        gripper_pos_tensor:torch.Tensor, n_voxels:int=20, sigma:float = 0.1,
        passive_geometry_mul:int=1,passive_softness_mul:int=1,
        device:str='cpu',
        **kwargs
    ):
        super(GeneratedPointEPCD, self).__init__(env)

        self.passive_geometry_mul = passive_geometry_mul
        self.passive_softness_mul = passive_softness_mul
        
        # region Preprocess and Attach Base
        gripper_pos_np = gripper_pos_tensor.detach().cpu().numpy()
        gripper_labels = self.get_muscle_label(gripper_pos_np)
        base_pos_tensor, base_labels = self.calculate_base_location(gripper_pos_np)
        
        complete_pos_tensor = torch.concat([gripper_pos_tensor,base_pos_tensor],dim=0)
        complete_labels = np.concatenate([gripper_labels,base_labels],axis=0)
        unique_lbls = np.unique(complete_labels)
        assert self.env.sim.solver.n_actuators == unique_lbls.shape[0], "The number of actuators must be equal to the number of generated labels. Probllem in configuration files"
        # endregion
        
        # region Calculate Actuator Direction
        all_part_pca_components, all_part_pca_singular_values, all_part_pc = extract_part_pca_inner(complete_pos_tensor.detach().cpu().numpy(),complete_labels,unique_lbls=set(complete_labels))
        actuator_directions = [] # TODO: make dict
        for k, part_pca_component in all_part_pca_components.items():
            # Taking the first PCA component as the direction for actuators
            actuator_directions.append(part_pca_component[0])
        actuator_directions = np.array(actuator_directions)
        actuator_directions = directions_to_spherical(actuator_directions)
        self.actuator_directions = nn.Parameter(torch.from_numpy(actuator_directions))
        # endregion
        
        # region Calculate Occupancy (!!!!! Ensure Differentiability)
        coords = self.env.design_space.get_x(s=0).float()
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
            
        coords_min, coords_mean, coords_max = coords.min(0), coords.mean(0), coords.max(0)
        points_min, points_mean, points_max = complete_pos_tensor.min(0), complete_pos_tensor.mean(0), complete_pos_tensor.max(0)
        def calibrate_points(_pts:torch.Tensor, y_offset=0.):
            _pts_calibrated = _pts - points_mean # center
            _pts_calibrated = _pts_calibrated / torch.max(points_max.values - points_min.values) * torch.max(coords_max.values - coords_min.values) # rescale
            _pts_calibrated = _pts_calibrated + coords_mean # recenter
            _pts_calibrated = _pts_calibrated + torch.clip(coords_min - _pts_calibrated.min(0).values, min=0, max=torch.inf) # make sure within min-bound
            _pts_calibrated = _pts_calibrated - torch.clip(_pts_calibrated.max(0).values - coords_max, min=0, max=np.inf) # make sure within max-bound
            _pts_calibrated[:,1] = _pts_calibrated[:,1] + y_offset # align lower bound in y-axis
            return _pts_calibrated
        complete_pos_tensor_calibrated = calibrate_points(complete_pos_tensor)
        y_offset = coords_min[1] - complete_pos_tensor_calibrated.min(0)[1] # The supposedly distance between the lowest point of the point cloud and the bouding box, to ensure it is at the bottom.
        
        pairwise_dist = torch.cdist(coords,complete_pos_tensor_calibrated) # , sim_coords(n), generated_coords(m)
        p_ji = torch.exp(-(pairwise_dist**2) / (2 * sigma**2))  # (n, m) # Smooth per-point contribution (Gaussian kernel)
        self.occupancy = 1 - torch.prod(1 - p_ji, dim=1)  # (n,) # Soft OR across points → final occupancy per voxel
        # endregion
        
        # region Cluster Sim Coords
        passive_lbl = 0
        self.actuator = torch.zeros((self.env.sim.solver.n_actuators, coords.shape[0])) #(n_act,n)
        self.is_passive = torch.zeros((coords.shape[0])) #(n)
        
        coords_lbls_prob = torch.zeros((self.env.sim.solver.n_actuators,coords.shape[0]))
        for lbl in unique_lbls:
            coords_cur_lbls_prob = p_ji[:,complete_labels==lbl]
            coords_lbls_prob[:,lbl] = 1 - torch.prod(1-coords_cur_lbls_prob,dim=1)
        coords_cluster = coords_lbls_prob.argmax(dim=1)
        for lbl in unique_lbls:
            if lbl==passive_lbl: self.is_passive[coords_cluster==lbl] = 1.
            else: self.actuator[lbl,coords_cluster==lbl] = 1.
        
        self.device = torch.device(device)
        self.to(self.device)
        
    def calculate_base_location(self,gripper_pos_np:np.ndarray):
        # Label 0 is reserved for the base, where no velocity is allowed to propagate to hold the gripper up.
        # The rest of the labels comes in pair (2n+1,2n+2) denotes the muscle pair n.
        
        base_pcd = o3d.io.read_point_cloud('diff_conditioning/simulation_env/asset/base_base.pcd')
        base_points = np.asarray(base_pcd.points)
        
        base_coords_min, base_coords_max,base_coords_mean = base_points.min(0),base_points.max(0),base_points.mean(0)
        gripper_points_min, gripper_points_max,gripper_points_mean = gripper_pos_np.min(0),gripper_pos_np.max(0),gripper_pos_np.mean(0)
        
        calibrated_base_pts = calibrate_points(base_points, mean = find_mid_lowest_pt(gripper_pos_np), scale=0.05*(gripper_points_max - gripper_points_min).sum() / (base_coords_max - base_coords_min).sum())
        base_labels = [0]*base_points.shape[0]
        
        return torch.from_numpy(calibrated_base_pts),np.array(base_labels)
    
    # region Muscle Label Generation
    def get_muscle_label(
        self,
        gripper_pos_np:np.ndarray,
        normalizing:bool=False,pca_components_index:List[int] = [],
        muscle_count:int = 16, 
        target_center:List[float] = [0.,1.,0.]
    ):
        assert muscle_count%2==0, "Muscle count must be an even number"
        
        points = gripper_pos_np
        points_std = StandardScaler().fit_transform(points)
        coarse_labels = self._generate_coarse_muscle(points_std,normalizing,pca_components_index,muscle_count)
        
        max_label = coarse_labels.max()
        target_center_np = np.array(target_center)
        additional_clusters = []
        for c in range(max_label + 1):
            cluster_idx = np.where(coarse_labels == c)[0]
            cluster_points = points_std[cluster_idx]
            if cluster_points.shape[0] == 0:
                continue
            
            additional_clusters += self._split_muscle_layers(cluster_points,target_center_np)
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
        muscle_count:int = 16
    ):  
        if normalizing:
            pca = PCA(n_components=3)
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
    
    def _split_muscle_layers(self, cluster_points:np.ndarray, target_center_np:np.ndarray):
        # PCA on cluster points
        cluster_center = cluster_points.mean(axis=0)
        pca_local = PCA(n_components=3)
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
    
    def reset(self):
        self.out_cache:Dict[str,Optional[torch.Tensor]] = dict(
            geometry=None, 
            softness=None, 
            actuator=None,
            actuator_direction=None,
            is_passive_fixed = None
        )
    
    def forward(self,inp=None):
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
    
    
# region Utilities
def find_mid_lowest_pt(points:np.ndarray,radius:float = 0.05):
    points_std = StandardScaler().fit_transform(points)
    mean_point = points_std.mean(0)
    dxz = points_std[:, [0, 2]] - mean_point[[0, 2]]
    dists_xz = np.linalg.norm(dxz, axis=1)
    mid_lowest_y = points[dists_xz < radius].min(0)[1]
    mid_lowest_pt = points[points[:, 1] == mid_lowest_y][0]
    return mid_lowest_pt

def calibrate_points(points:np.ndarray,mean:np.ndarray,flipped_x:bool = False,flipped_y:bool = False,flipped_z:bool=False,scale=1.0):
    points_mean = points.mean(0)
    norm_points = points - points_mean
    norm_points = norm_points * scale
    if flipped_x: norm_points[:,0] = -norm_points[:,0]
    if flipped_y: norm_points[:,1] = -norm_points[:,1]
    if flipped_z: norm_points[:,2] = -norm_points[:,2]
    points_calibrated = norm_points + mean
    return points_calibrated
# endregion