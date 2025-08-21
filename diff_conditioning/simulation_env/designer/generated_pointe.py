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

from .base import Base
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
        
    def calculate_base_location(self,gripper_pos_np:np.ndarray):
        # Label 0 is reserved for the base, where no velocity is allowed to propagate to hold the gripper up.
        # The rest of the labels comes in pair (2n+1,2n+2) denotes the muscle pair n.
        
        base_pcd = o3d.io.read_point_cloud('diff_conditioning/simulation_env/asset/fixed_base.pcd')
        base_points = np.asarray(base_pcd.points)
        
        base_coords_min, base_coords_max = base_points.min(0),base_points.max(0)
        gripper_points_min, gripper_points_max = gripper_pos_np.min(0),gripper_pos_np.max(0)
        base_extent = base_coords_max - base_coords_min
        gripper_extent = gripper_points_max - gripper_points_min
        gripper_base_scale = gripper_extent/base_extent
        
        lowest_pt = find_mid_lowest_pt(gripper_pos_np,radius=0.25)
    
        calibrated_base_pts = calibrate_points(
            base_points, 
            mean = lowest_pt, 
            scale=np.array([
                0.1*gripper_base_scale[0],
                0.02*gripper_base_scale[1],
                0.1*gripper_base_scale[2]
            ])
        )
    
        base_labels = [0]*base_points.shape[0]
        
        return torch.from_numpy(calibrated_base_pts).to(self.device),np.array(base_labels)
    
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
        coarse_labels = self._generate_coarse_muscle(points_std,normalizing,pca_components_index,muscle_count,feature_selector)
        
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
    
    def _split_muscle_layers(self, cluster_points:np.ndarray, target_center_np:np.ndarray):
        # PCA on cluster points
        cluster_center = cluster_points.mean(axis=0)
        # TODO: Fix problem here sometimes num samples < 3
        if min(*cluster_points.shape,3)<3: print("Warning!!!: Numer of points in cluster is small <3.")
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
    
    def forward(self,inp:torch.Tensor):
        inp = inp.to(self.device)
        
        self.create_representation_from_tensor(inp)
        
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
    
    def create_representation_from_tensor(self,gripper_pos_tensor:torch.Tensor):
        # region Preprocess and Attach Base
        gripper_pos_np = gripper_pos_tensor.detach().cpu().numpy() # [N,C]
        gripper_labels = self.get_muscle_label(gripper_pos_np)
        base_pos_tensor, base_labels = self.calculate_base_location(gripper_pos_np)
        
        complete_pos_tensor = torch.concat([base_pos_tensor,gripper_pos_tensor],dim=0).float()
        complete_labels = np.concatenate([base_labels,gripper_labels],axis=0)
    
        # visualize_point_cloud(complete_pos_tensor,complete_labels)
        unique_lbls = np.unique(complete_labels)
        if not self.env.sim.solver.n_actuators == unique_lbls.shape[0]:
            print("Warning!!!!: \n The number of actuators must be equal to the number of generated labels. Probllem in configuration files")
        # endregion
        
        # region Calculate Actuator Direction
        all_part_pca_components, all_part_pca_singular_values, all_part_pc = extract_part_pca_inner(complete_pos_tensor.detach().cpu().numpy(),complete_labels,unique_lbls=range(self.env.sim.solver.n_actuators))        
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
        points_min = complete_pos_tensor.min(0).values if self.bounding_box is None or 'min' not in self.bounding_box else torch.min(torch.tensor(self.bounding_box['min']),complete_pos_tensor.min(0).values)
        points_mean = complete_pos_tensor.mean(0) if self.bounding_box is None or 'mean' not in self.bounding_box else torch.tensor(self.bounding_box['mean'])
        points_max = complete_pos_tensor.max(0).values if self.bounding_box is None or 'max' not in self.bounding_box else torch.max(torch.tensor(self.bounding_box['max']),complete_pos_tensor.max(0).values)

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
            coords_cur_lbls_prob = p_ji[:,complete_labels==lbl]
            occupancy_cluster = (1 - torch.prod(1-coords_cur_lbls_prob,dim=1)) #*(1. if lbl!=passive_lbl else 1e4)
            coords_lbls_prob[:,lbl] = occupancy_cluster
        coords_cluster = coords_lbls_prob.argmax(dim=1)
        for lbl in unique_lbls:
            if lbl==passive_lbl: self.is_passive[coords_cluster==lbl] = 1.
            else: self.actuator[lbl,coords_cluster==lbl] = 1.
        
        # print(self.is_passive[(self.is_passive + self.occupancy)>1.].sum())
        # endregion
        

# region Utilities
def find_mid_lowest_pt(points:np.ndarray,radius:float = 0.05):
    points_std = StandardScaler().fit_transform(points)
    mean_point = points_std.mean(0)
    dxz = points_std[:, [0, 2]] - mean_point[[0, 2]]
    dists_xz = np.linalg.norm(dxz, axis=1)
    mid_lowest_y = points[dists_xz < radius].min(0)[1]
    mid_lowest_pt = points[points[:, 1] == mid_lowest_y][0]
    return mid_lowest_pt

def calibrate_points(points:np.ndarray,mean:np.ndarray,flipped_x:bool = False,flipped_y:bool = False,flipped_z:bool=False,scale:Union[float,np.ndarray]=1.0):
    points_mean = points.mean(0)
    norm_points = points - points_mean
    norm_points = norm_points * scale
    if flipped_x: norm_points[:,0] = -norm_points[:,0]
    if flipped_y: norm_points[:,1] = -norm_points[:,1]
    if flipped_z: norm_points[:,2] = -norm_points[:,2]
    points_calibrated = norm_points + mean
    return points_calibrated


def visualize_point_cloud(pcd_coords,labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_coords.detach().cpu().numpy())
    
    disc_colors = np.array(plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors)
    disc_colors = np.concatenate([np.array([[0,0,0]]),disc_colors],axis=0)
    colors = disc_colors[labels]
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    all_part_pca_components, all_part_pca_singular_values, all_part_pc, color_debug = extract_part_pca(pcd, return_part_colors=True)
    line_meshes = []
    for i, (k, part_pca_components) in enumerate(all_part_pca_components.items()):
        start = all_part_pc[k].mean(0)
        end1 = start + part_pca_components[0] * all_part_pc[k].std(0)[0]
        end2 = start + part_pca_components[1] * all_part_pc[k].std(0)[1]
        end3 = start + part_pca_components[2] * all_part_pc[k].std(0)[2]
        line_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1],]
        for end_i, end in enumerate([end1, end2, end3]):
            line_mesh = get_arrow(start, end)
            line_mesh.paint_uniform_color(line_colors[end_i])
            line_meshes.append(line_mesh)

    o3d.visualization.draw_geometries([pcd] + line_meshes)
# endregion