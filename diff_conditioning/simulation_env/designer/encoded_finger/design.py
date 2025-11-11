import torch
import torch.nn as nn
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import numpy as np

import open3d as o3d
from pytorch3d.transforms import quaternion_multiply,quaternion_apply,axis_angle_to_quaternion

from typing import TYPE_CHECKING, Dict, List, Optional, Callable, Literal,Union, Tuple
from typing_extensions import TypedDict
import logging

from ..base import Base
from .base_config import Config
from ..utils import calibrate_translate_pts,rotate_y,visualize_point_cloud
from .design_loss_func import finger_penetration_loss

from softzoo.utils.general_utils import extract_part_pca_inner,row_permutation,extract_part_pca
from softzoo.utils.computation_utils import directions_to_spherical

if TYPE_CHECKING:
    from softzoo.envs.base_env import BaseEnv

class EncodedFinger(Base):
    # region Initialization
    def __init__(
        self, base_config:Config, env:'BaseEnv', 
        sigma:float = 7e-4,
        passive_geometry_mul:int=1,passive_softness_mul:int=1,
        device:str='cpu',
        bounding_box:Optional[Dict[Literal['max','mean','min'],List[float]]] = None,
    ):
        super(EncodedFinger,self).__init__(env)
        
        self.passive_geometry_mul = passive_geometry_mul
        self.passive_softness_mul = passive_softness_mul
        self.base_config = base_config
        self.bounding_box = bounding_box
        self.sigma = sigma
        
        self.device = torch.device(device)
        self.to(self.device)
        
        self.original_coords = self.env.design_space.get_x(s=0).float()
        if isinstance(self.original_coords, np.ndarray):
            self.original_coords = torch.from_numpy(self.original_coords).float()
        self.original_coords = self.original_coords.to(self.device)
        
        self._load_base_segment()
        self._load_fixed_base()
    def _load_base_segment(self):
        base_pcd = o3d.io.read_point_cloud(self.base_config['segment_config']['complete_segment_path'])
        base_pts = np.asarray(base_pcd.points)
        base_colors = np.asarray(base_pcd.colors)

        cylinder_mask = np.all(base_colors==self.base_config['segment_config']['cylinder_color'],axis=1)
        self.cylinder_pts = torch.from_numpy(base_pts[cylinder_mask]).to(self.device).float() #(N,3)
        self.conn_ends = torch.from_numpy(base_pts[~cylinder_mask]).to(self.device).float() # (M,3)
        self.cylinder_num_pts = self.cylinder_pts.shape[0]
        self.conn_ends_num_pts = self.conn_ends.shape[0]
    def _load_fixed_base(self):
        base_pcd = o3d.io.read_point_cloud(self.base_config['fixed_base_config']['fixed_base_path'])
        base_points = torch.from_numpy(np.asarray(base_pcd.points)).to(self.device).float()
        calibrated_base_pts = calibrate_translate_pts(
            base_points, 
            mean = torch.tensor([0.,0.,0.]).to(self.device), 
            scale=torch.tensor([1.,1.,1.]).to(self.device)
        )
        base_labels = [0 for _ in range(base_points.shape[0])]
        self.base_pts = calibrated_base_pts
        self.base_lbls = np.array(base_labels)
    # endregion    

    # region Misc
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
        self.design_loss = torch.zeros(1,device=self.device)
    # endregion
    
    def forward(self,ctrl_tensor:torch.Tensor,end_prob_mask:torch.Tensor):
        occupancy, actuator,is_passive,softness,suction_val,actuator_directions = self._create_representation_from_tensor(ctrl_tensor,end_prob_mask)
        active_geometry = occupancy
        passive_geometry = occupancy * self.passive_geometry_mul # NOTE: the same as active
        geometry = torch.where(is_passive==1., passive_geometry, active_geometry)
        
        active_softness = softness
        passive_softness = softness * self.passive_softness_mul # NOTE: the same as active
        softness_scaled = torch.where(is_passive==1., passive_softness, active_softness)
        
        self.out_cache['geometry'] = geometry
        self.out_cache['softness'] = softness_scaled
        self.out_cache['actuator'] = actuator
        self.out_cache['actuator_direction'] = actuator_directions
        self.out_cache['is_passive_fixed'] = is_passive.to(torch.float32)
        self.out_cache['suction_val'] = suction_val
        
        # NOTE: must use clone here otherwise tensor may be modified in-place in sim
        design = {k: v.clone() for k, v in self.out_cache.items()}
        return design, self.design_loss

    def _create_representation_from_tensor(self, ctrl_tensor:torch.Tensor,end_prob_mask:torch.Tensor):
        """
        Generate gripper with fingers determined by the ctrl_tensor
        Args:
            ctrl_tensor (torch.Tensor): Control points tensor of shape (num_finger, num_segment, D).
            [
                [segment1, segment2,... segmentN],
                [segment1, segment2,... segmentN]
            ]
        """
        ctrl_tensor = ctrl_tensor.to(self.device)
        end_prob_mask = end_prob_mask.to(self.device)
        processed_end_prob_mask = self._filter_segment_encoding(end_prob_mask)
        # processed_end_prob_mask = torch.ones_like(processed_end_prob_mask)
        print(processed_end_prob_mask.sum(dim=1))
        filtered_ctrl_tensor = ctrl_tensor * processed_end_prob_mask[:,:,None]
        
        num_fingers = filtered_ctrl_tensor.shape[0]
        num_segment_per_finger = filtered_ctrl_tensor.shape[1]
        
        complete_pos_tensor = self._create_gripper(
            num_fingers,num_segment_per_finger,
            filtered_ctrl_tensor,processed_end_prob_mask
        ).float()
        
        complete_labels_np = self._create_labels(processed_end_prob_mask)
        # visualize_point_cloud(complete_pos_tensor,complete_labels_np)
        # print(complete_pos_tensor.max(0),complete_pos_tensor.min(0))
        unique_lbls = np.unique(complete_labels_np)
        if not self.env.sim.solver.n_actuators == unique_lbls.shape[0]:
            logging.warning(f"Warning!!!!: \n The number of actuators {self.env.sim.solver.n_actuators} must be equal to the number of generated labels {unique_lbls.shape[0]}. Probllem in configuration files")
            if unique_lbls.shape[0] > self.env.sim.solver.n_actuators:
                raise NotImplementedError(f"The number of actuators {unique_lbls.shape[0]} exceed the limit of the simulation {self.env.sim.solver.n_actuators}.")
        
        actuator_directions = self._get_actuator_direction(complete_labels_np,complete_pos_tensor)
        softness_mul_by_lbl = self._get_sim_softness(filtered_ctrl_tensor)
        suction_mul_by_lbl  = self._get_sim_suctions(filtered_ctrl_tensor)
        occupancy, actuator,is_passive,softness,suction_val = self._get_sim_occupancy(
            complete_pos_tensor,complete_labels_np,
            unique_lbls,
            softness_mul_by_lbl,suction_mul_by_lbl
        )
        
        return occupancy, actuator,is_passive,softness,suction_val, actuator_directions
    
    def _filter_segment_encoding(
        self,
        end_prob_mask:torch.Tensor, threshold:float = 0.25
    ):
        end_prob_softmax = torch.softmax(end_prob_mask,dim=1)
        end_prob_cum_sum = torch.cumsum(end_prob_softmax,dim=1)
        end_prob_threshold = 1.-end_prob_cum_sum
        end_prob_binaries = torch.where(
            end_prob_threshold > threshold,
            torch.ones_like(end_prob_threshold),
            torch.zeros_like(end_prob_threshold)
        )
        end_prob_pseudo_flow = end_prob_binaries - end_prob_threshold.detach() + end_prob_threshold
        return end_prob_pseudo_flow
    
    def _create_gripper(
        self,
        num_fingers:int,num_segment_per_finger:int,
        ctrl_tensor:torch.Tensor,end_prob_mask:torch.Tensor
    )->torch.Tensor:
        """
        Generate gripper with fingers determined by the ctrl_tensor
        Args:
            ctrl_tensor (torch.sor): Control points tensor of shape (num_finger, num_segment, D).
            end_prob_mask (torch.Tensor): The differentiable binary mask to represent the cutoff points. (num_points,num_segment)
        Returns:
            torch.Tensor: Transform the ctrl_tensor into a working gripper with custom fingers.
        """
        
        splined_cylinder_pts = self._transform_splining(num_fingers,num_segment_per_finger,ctrl_tensor) # (num_finger,num_seg,N,3)
        full_segment_pts,top_conn_pts = self._transform_lengthening(num_fingers,num_segment_per_finger,ctrl_tensor,splined_cylinder_pts) # (num_finger,num_seg,N+M,3)
        rotated_segments = self._rotate_segment(
            num_fingers,num_segment_per_finger,
            ctrl_tensor,full_segment_pts,top_conn_pts
        ) # (num_finger,num_seg,N+M,3)
        
        full_fingers = rotated_segments.reshape(num_fingers,-1,3)
        reversed_finger = full_fingers * torch.tensor([[[1.,-1.,1.]]]).to(self.device) # (num_finger,num_segment*(N+M),3)
        
        # Plug fingers into the base, the root of the finger is already at (0,0,0)
        angles = torch.arange(num_fingers, device=self.device) * (2 * torch.pi / num_fingers) #(num_fingers)
        quats_angles = torch.stack([
            torch.cos(angles/2),
            torch.zeros_like(angles),
            torch.sin(angles/2),
            torch.zeros_like(angles)
        ],dim=-1) # (num_fingers,4)
        
        x_axis_pos = torch.cos(torch.pi - angles) * self.base_config['fixed_base_config']['finger_radius']
        z_axis_pos = torch.sin(torch.pi - angles) * self.base_config['fixed_base_config']['finger_radius']
        y_axis_pos = torch.zeros_like(x_axis_pos)
        fingers_pos = torch.stack([x_axis_pos, y_axis_pos, z_axis_pos], dim=1)  # [num_finger,3]
        # print(fingers_pos)
        # print(self.base_pts.max(0),self.base_pts.min(0))
        
        oriented_finger = quaternion_apply(
            quats_angles[:,None,:],     # (num_fingers,1,4)
            reversed_finger             # (num_fingers,num_segments*(N+M),3)
        )
        translated_oriented_finger = oriented_finger + fingers_pos[:,None,:] # (num_fingers,num_segments*(N+M),3)
        self.design_loss += finger_penetration_loss(translated_oriented_finger.float())
        filtered_finger = translated_oriented_finger.reshape(num_fingers,num_segment_per_finger,-1,3)[end_prob_mask.bool()] #(filtered_segment,(N+M),3)
        
        return torch.concat([self.base_pts,filtered_finger.reshape(-1,3)],dim = 0) # (num_particle,3)
    
    def _create_labels(self,end_probs_binary:torch.Tensor):
        # Scale the connection ends to correspond with new length
        top_conn_ends = self.conn_ends[self.conn_ends[:,1]>0] # (M1,3)
        bottom_conn_ends = self.conn_ends[self.conn_ends[:,1]<0] # (M2,3)
        full_segment_pts =  torch.cat([
            self.cylinder_pts,
            top_conn_ends,
            bottom_conn_ends
        ],dim=0) # (N+M,3)
        
        segment_lbls = torch.ones(full_segment_pts.shape[0],dtype=torch.int8)    #(N+M,)
        segment_lbls[full_segment_pts[:,0]>0]=2                 #(N+M,)
        
        total_segments = end_probs_binary.sum().item()
        
        repeated_segment = segment_lbls.repeat(int(total_segments),1)    # (total_segment,N+M)
        offsets = torch.arange(total_segments)[:,None].int() * 2          # (total_segment,N+M)
        offseted_segment_lbl = repeated_segment + offsets                   # (num_finger,num_segment,N+M)
        
        return np.concatenate([self.base_lbls,offseted_segment_lbl.flatten().detach().cpu().numpy()],axis=-1)
    
    def _get_actuator_direction(self,complete_labels_np:np.ndarray,complete_pos_tensor:torch.Tensor,):
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
        return torch.from_numpy(actuator_directions).to(self.device)
        # endregion
   
    def _get_sim_occupancy(
        self,
        complete_pos_tensor:torch.Tensor,complete_labels_np:np.ndarray,
        unique_lbls:np.ndarray,
        softness_mul:torch.Tensor,suction_mul:torch.Tensor
    ):
        """
        Calculate all metrics related to the simulation representation
        Args:
            complete_pos_tensor: The generated representation of the gripper that needs to be ported to the simulation - (num_particle,3)
            complete_labels_np: The np.ndarray of all labels of all particles in the gripper - (num_particles,)
            unique_lbls: Set of unique labels
            softness_mul: The softness multiplier for each label.
            suction_mul: The suction multiplier for each label
        """
        # region Calculate Occupancy (!!!!! Ensure Differentiability)
        coords_min, coords_mean, coords_max = self.original_coords.min(0).values, self.original_coords.mean(0), self.original_coords.max(0).values
        actual_min,actual_mean,actual_max = complete_pos_tensor.min(0).values,complete_pos_tensor.mean(0),complete_pos_tensor.max(0).values 
        
        points_min = actual_min if self.bounding_box is None or 'min' not in self.bounding_box else torch.min(torch.tensor(self.bounding_box['min']).to(self.device),actual_min)
        points_mean =  actual_mean if self.bounding_box is None or 'mean' not in self.bounding_box else torch.tensor(self.bounding_box['mean']).to(self.device)
        points_max = actual_max if self.bounding_box is None or 'max' not in self.bounding_box else torch.max(torch.tensor(self.bounding_box['max']).to(self.device),actual_max)

        points_bb = points_max - points_min
        coords_bb = coords_max - coords_min
        scale_diff = coords_bb / points_bb

        def calibrate_points(_pts:torch.Tensor, y_offset=0.):
            _pts_calibrated = _pts - points_mean # center
            _pts_calibrated = _pts_calibrated * torch.min(scale_diff) * 0.99 # rescale
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
        occupancy = 1 - torch.prod(1 - p_ji, dim=1)  # (n,) # Soft OR across points → final occupancy per voxel
        # endregion
        
        # region Cluster Sim Coords
        passive_lbl = 0
        actuator = torch.zeros((self.env.sim.solver.n_actuators, self.original_coords.shape[0])).to(self.device) #(n_act,n)
        is_passive = torch.zeros((self.original_coords.shape[0])).bool().to(self.device) #(n)
        softness = torch.zeros((self.original_coords.shape[0])).to(self.device)
        suction_val = torch.zeros((self.original_coords.shape[0])).to(self.device)
        
        coords_lbls_prob = torch.zeros((self.original_coords.shape[0],self.env.sim.solver.n_actuators)).to(self.device)
        for lbl in unique_lbls:
            coords_cur_lbls_prob = p_ji[:,complete_labels_np==lbl]
            occupancy_cluster = (1 - torch.prod(1-coords_cur_lbls_prob,dim=1)) 
            coords_lbls_prob[:,lbl] = occupancy_cluster
        coords_cluster = coords_lbls_prob.argmax(dim=1)
        for lbl in unique_lbls:
            if lbl==passive_lbl: is_passive[coords_cluster==lbl] = 1.
            else: actuator[lbl,coords_cluster==lbl] = 1.
            softness[coords_cluster==lbl] = softness_mul[lbl]
            suction_val[coords_cluster==lbl] = suction_mul[lbl]
        # endregion
        
        return occupancy,actuator,is_passive,softness,suction_val
    
    def _get_sim_softness(self,ctrl_tensor:torch.Tensor):
        softness_mul = ctrl_tensor[:,:,8].flatten() # (num_finger*num_segments)
        softness_mul_scaled = softness_mul * (
            self.base_config['segment_config']['softness_range'][1] - self.base_config['segment_config']['softness_range'][0]
        ) + self.base_config['segment_config']['softness_range'][0] # (B,)
        softness_mul_by_lbls = softness_mul_scaled.repeat_interleave(2)
        return torch.concat([
            torch.tensor([1.]).to(self.device),
            softness_mul_by_lbls
        ])
    
    def _get_sim_suctions(self,ctrl_tensor:torch.Tensor):
        suction_mul = ctrl_tensor[:,:,10]
        suction_mul_scaled = suction_mul * (
            self.base_config['segment_config']['suction_range'][1] - self.base_config['segment_config']['suction_range'][0]
        ) + self.base_config['segment_config']['suction_range'][0] # (B,)
        suction_mul_by_lbls = suction_mul_scaled.repeat_interleave(2)
        return torch.concat([
            torch.tensor([0.]).to(self.device),
            suction_mul_by_lbls
        ])
    
    # region Finger Transformation
    def _transform_splining(
        self,
        num_fingers:int,num_segments:int,
        ctrl_tensor:torch.Tensor
    )->torch.Tensor:
        """
        Transform the segment thickness through splining
        Args:
            num_fingers (int): Number of fingers.
            num_segments (int): Number of segments per finger.
            ctrl_tensor (torch.Tensor): Control points tensor of shape (num_finger,num_seg,D).
        Returns:
            torch.Tensor: Modified cylinder points after spline deformation of shape (num_finger,num_seg,N,3)
        """
        # B = num_fingers * num_segments
        # N = num_pts_per_segment
        t_sample = self.cylinder_pts[:,1] + (self.base_config['segment_config']['base_length']/2)
        
        spline_ctrl_tensor = ctrl_tensor[:,:,1:5].reshape(num_fingers*num_segments,4) # (num_fingers*num_seg,4)
        spline_ctrl_pts = spline_ctrl_tensor*(
            self.base_config['segment_config']['spline_range'][1]-self.base_config['segment_config']['spline_range'][0]
        ) + self.base_config['segment_config']['spline_range'][0] # (B,4)
        outer_cols = torch.ones(spline_ctrl_pts.shape[0],2).to(self.device)
        full_ctrl_pts = torch.cat([
            outer_cols[:,:1], # (B,1)
            spline_ctrl_pts,  # (B,4)
            outer_cols[:,1:]  # (B,1)
        ],dim=1) # (B,6)
        
        t = torch.linspace(0,self.base_config['segment_config']['base_length'],6).to(self.device)
        coeffs = natural_cubic_spline_coeffs(t, full_ctrl_pts.T) # [6,(6,B)]
        spline = NaturalCubicSpline(coeffs)
        spline_res = spline.evaluate(t_sample) # (N,B)
        spline_res = spline_res.T.unsqueeze(-1) # (B,N,1)
        
        mod_cylinder_pts = self.cylinder_pts.repeat(spline_res.shape[0],1,1) # (B,N,3)
        spline_scale = torch.cat([
            spline_res,
            torch.ones_like(spline_res),
            spline_res
        ],dim=-1) # (B,N,3)
        return (mod_cylinder_pts * spline_scale).reshape(num_fingers,num_segments,self.cylinder_num_pts,3)
    
    def _transform_lengthening(
        self,
        num_fingers:int,num_segments:int,
        ctrl_tensor:torch.Tensor,batched_cylinder_pts:torch.Tensor
    )->Tuple[torch.Tensor,torch.Tensor]:
        """
        Transform the segment through lengthening
        Args:
            num_fingers (int): Number of fingers.
            num_segments (int): Number of segments per finger.
            ctrl_tensor (torch.Tensor): Control points tensor of shape (num_finger,num_seg,D).
            batched_cylinder_pts (torch.Tensor): Cylinder points tensor of shape (num_finger,num_seg,N,3).
        Returns:
            torch.Tensor: Modified full segment points after lengthening deformation of shape (num_finger,num_seg,N+M,3)
        """
        # B = num_fingers * num_segments
        # N = num_pts_per_segment
        lengthen_tensor = ctrl_tensor[:,:,0].reshape(num_fingers*num_segments) # (B,)
        
        lengthen_val = lengthen_tensor * (
            self.base_config['segment_config']['lengthen_range'][1]-self.base_config['segment_config']['lengthen_range'][0]
        ) + self.base_config['segment_config']['lengthen_range'][0] # (B,)
        lengthen_val_reshaped = lengthen_val[:,None,None ]# (B,1,1)
        lengthen_scale = torch.cat([
            torch.ones_like(lengthen_val_reshaped),
            lengthen_val_reshaped,
            torch.ones_like(lengthen_val_reshaped)
        ],dim=-1) # (B,N,3)
        batched_mod_cylinder_pts = batched_cylinder_pts.reshape(num_fingers*num_segments,self.cylinder_num_pts,3) * lengthen_scale # (B,N,3)
        
        # Scale the connection ends to correspond with new length
        top_conn_ends = self.conn_ends[self.conn_ends[:,1]>0] # (M1,3)
        bottom_conn_ends = self.conn_ends[self.conn_ends[:,1]<0] # (M2,3)
        
        offset_length = (self.base_config['segment_config']['base_length']/2 * (lengthen_val - 1.0))[:,None,None] # (B,1,1)
        top_conn_offset = torch.cat([
            torch.zeros_like(offset_length),
            offset_length,
            torch.zeros_like(offset_length)
        ],dim=-1)
        bottom_conn_offset = torch.cat([
            torch.zeros_like(offset_length), 
            -offset_length,
            torch.zeros_like(offset_length)
        ],dim=-1)
        translated_top_conn_ends = top_conn_ends.repeat(num_fingers*num_segments,1,1) + top_conn_offset # (B,M1,3)
        translated_bottom_conn_ends = bottom_conn_ends.repeat(num_fingers*num_segments,1,1) + bottom_conn_offset # (B,M2,3)
        
        # The list of origin-centered segment
        complete_origin_centered_segment =  torch.cat([
            batched_mod_cylinder_pts,
            translated_top_conn_ends,
            translated_bottom_conn_ends
        ],dim=1) # (B,N+M,3)
        
        # Shift the segment so that the bottom connection ends is centered at origin
        #           |   |
        #           |   |
        # |   |     |   |
        # |   |     |   |
        # -----  => -----
        # |   |
        # |   |
        total_offset_y = (self.base_config['segment_config']['base_length']/2 * lengthen_val + self.base_config['segment_config']['radius'])[:,None,None] # (B,1,1)
        total_offset = torch.cat([
            torch.zeros_like(total_offset_y),
            total_offset_y,
            torch.zeros_like(total_offset_y)
        ],dim=-1) # (B,1,3)
        
        # Get top connection points
        top_conn_y = self.base_config['segment_config']['radius'] + self.base_config['segment_config']['base_length']*lengthen_val + self.base_config['segment_config']['radius'] #(B,)
        top_conn_pts = torch.stack([
            torch.zeros_like(top_conn_y),
            top_conn_y,
            torch.zeros_like(top_conn_y)
        ],dim=-1) # (B,3)
        
        return (
            (complete_origin_centered_segment + total_offset).reshape(num_fingers,num_segments,self.cylinder_num_pts+self.conn_ends_num_pts,3), # (num_finger,num_seg,N+M,3)
            top_conn_pts.reshape(num_fingers,num_segments,3) # (num_finger,num_seg,3)
        )
        
    def _rotate_segment(
        self,
        num_fingers:int,num_segments:int,
        ctrl_tensor:torch.Tensor,
        full_segment_pts:torch.Tensor, top_conn_pts:torch.Tensor
    )->torch.Tensor:
        """
        Rotate each segment to the desired orientation
        Args:
            num_fingers (int): Number of fingers.
            num_segments (int): Number of segments per finger.
            ctrl_tensor (torch.Tensor): Control points tensor of shape (num_finger,num_seg,D).
            full_segment_pts (torch.Tensor): Full segment points tensor of shape (num_finger,num_seg,N+M,3).
            top_conn_pts (torch.Tensor): Top connection points tensor of shape (num_finger,num_seg,3).
        Returns: 
        """
        axis_angle_tensor = ctrl_tensor[:,:,5:8] # (num_finger,num_seg,3)
        rotation_range = torch.tensor(self.base_config['segment_config']['rotation_range']).to(self.device) # (3,2)
        axis_angle_scaled = axis_angle_tensor*(rotation_range[None,None,:,1] - rotation_range[None,None,:,0]) + rotation_range[None,None,:,0] # (num_finger,num_seg, 3)
        
        quat_tensor = axis_angle_to_quaternion(axis_angle_scaled)
        quat_normalized = quat_tensor / quat_tensor.norm(dim=-1,keepdim=True) # (num_finger,num_seg,4)
        cummulative_quat_lst = [quat_normalized[:,0,:]]  # list of (num_finger,4)
        for i in range(1,num_segments):
            cummulative_quat_lst.append(quaternion_multiply(cummulative_quat_lst[-1],quat_normalized[:,i,:]))
        cummulative_quat = torch.stack(cummulative_quat_lst,dim=1)  # (num_finger, num_segment, 4)
        oriented_segments = quaternion_apply(
            cummulative_quat[:,:,None,:],   # (num_finger,num_seg,1,4)
            full_segment_pts,               # (num_finger,num_seg,N+M,3)
        )
        
        rotated_top_conn_pts = quaternion_apply(
            cummulative_quat,               # (num_finger,num_seg,4)
            top_conn_pts,                   # (num_finger,num_seg,3)
        )
        translated_rotated_top_conn_pts = torch.cumsum(rotated_top_conn_pts, dim=1).clone()
        rolled_offset = torch.roll(translated_rotated_top_conn_pts,shifts=1,dims=1)
        rolled_offset[:,0,:] = torch.zeros_like(rolled_offset[:,0,:]) # (num_finger,num_seg,3)
        
        return oriented_segments + rolled_offset[:,:,None,:] # (num_finger,num_seg,N+M,3)
    # endregion
    
        