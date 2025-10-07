import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import numpy as np

import open3d as o3d
from pytorch3d.transforms import quaternion_multiply,quaternion_apply,axis_angle_to_quaternion

from typing import TYPE_CHECKING, Dict, List, Optional, Callable, Literal,Union, Tuple
from typing_extensions import TypedDict

from ..base import Base
from .base_config import Config
from ..utils import calibrate_translate_pts,rotate_y

if TYPE_CHECKING:
    from softzoo.envs.base_env import BaseEnv

# class EncodedFinger(Base):
class EncodedFinger(torch.nn.Module):
    def __init__(
        self, base_config:Config, #env:'BaseEnv', 
        sigma:float = 7e-4,
        passive_geometry_mul:int=1,passive_softness_mul:int=1,
        device:str='cpu',
        bounding_box:Optional[Dict[Literal['max','mean','min'],List[float]]] = None,
    ):
        super(EncodedFinger,self).__init__()
        
        self.passive_geometry_mul = passive_geometry_mul
        self.passive_softness_mul = passive_softness_mul
        self.base_config = base_config
        self.bounding_box = bounding_box
        self.sigma = sigma
        
        self.device = torch.device(device)
        self.to(self.device)
        
        # self.original_coords = self.env.design_space.get_x(s=0).float()
        # if isinstance(self.original_coords, np.ndarray):
        #     self.original_coords = torch.from_numpy(self.original_coords).float()
        # self.original_coords = self.original_coords.to(self.device)
        
        self._load_base_segment()
        self._load_fixed_base()
    
    def _load_base_segment(self):
        base_pcd = o3d.io.read_point_cloud(self.base_config['segment_config']['complete_segment_path'])
        base_pts = np.asarray(base_pcd.points)
        base_colors = np.asarray(base_pcd.colors)

        cylinder_mask = np.all(base_colors==self.base_config['segment_config']['cylinder_color'],axis=1)
        self.cylinder_pts = torch.from_numpy(base_pts[cylinder_mask]).to(self.device) #(N,3)
        self.conn_ends = torch.from_numpy(base_pts[~cylinder_mask]).to(self.device) # (M,3)
        self.cylinder_num_pts = self.cylinder_pts.shape[0]
        self.conn_ends_num_pts = self.conn_ends.shape[0]
    def _load_fixed_base(self):
        base_pcd = o3d.io.read_point_cloud(self.base_config['fixed_base_config']['fixed_base_path'])
        base_points = torch.from_numpy(np.asarray(base_pcd.points)).to(self.device)
        calibrated_base_pts = calibrate_translate_pts(
            base_points, 
            mean = torch.tensor([0.,0.,0.]).to(self.device), 
            scale=torch.tensor([1.,1.,1.]).to(self.device)
        )
        base_labels = [0 for _ in range(base_points.shape[0])]
        self.base_pts = calibrated_base_pts
        self.base_lbls = np.array(base_labels)    

    
        
    def forward(self, ctrl_tensor:torch.Tensor)->torch.Tensor:
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
        gripper_pos_tensor = self._create_gripper(ctrl_tensor)
        return gripper_pos_tensor
    
    def _create_gripper(self,ctrl_tensor:torch.Tensor)->torch.Tensor:
        """
        Generate gripper with fingers determined by the ctrl_tensor
        Args:
            ctrl_tensor (torch.Tensor): Control points tensor of shape (num_finger, num_segment, D).
        Returns:
            torch.Tensor: Transform the ctrl_tensor into a working gripper with custom fingers.
        """
        num_fingers = ctrl_tensor.shape[0]
        num_segment_per_finger = ctrl_tensor.shape[1]
        
        splined_cylinder_pts = self._transform_splining(num_fingers,num_segment_per_finger,ctrl_tensor) # (num_finger,num_seg,N,3)
        full_segment_pts,top_conn_pts = self._transform_lengthening(num_fingers,num_segment_per_finger,ctrl_tensor,splined_cylinder_pts) # (num_finger,num_seg,N+M,3)
        rotated_segments = self._rotate_segment(
            num_fingers,num_segment_per_finger,
            ctrl_tensor,full_segment_pts,top_conn_pts
        ) # (num_finger,num_seg,N+M,3)
        full_fingers = rotated_segments.reshape(num_fingers,-1,3)
        reversed_finger = full_fingers * torch.tensor([[[1.,-1.,1.]]]) # (num_finger,num_segment*(N+M),3)
        
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
        
        oriented_finger = quaternion_apply(
            quats_angles[:,None,:],     # (num_fingers,1,4)
            reversed_finger             # (num_fingers,(N+M)*num_segments,3)
        )
        translated_oriented_finger = oriented_finger + fingers_pos[:,None,:]
        
        return torch.concat([self.base_pts,translated_oriented_finger.reshape(-1,3)],dim = 0) # (num_particle,3)
        
        
    
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
        outer_cols = torch.ones(spline_ctrl_pts.shape[0],2)
        full_ctrl_pts = torch.cat([
            outer_cols[:,:1], # (B,1)
            spline_ctrl_pts,  # (B,4)
            outer_cols[:,1:]  # (B,1)
        ],dim=1) # (B,6)
        
        t = torch.linspace(0,self.base_config['segment_config']['base_length'],6)
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
        print(axis_angle_scaled)
        
        quat_tensor = axis_angle_to_quaternion(axis_angle_scaled)
        quat_normalized = quat_tensor / quat_tensor.norm(dim=-1,keepdim=True) # (num_finger,num_seg,4)
        cummulative_quat = quat_normalized.clone() # (num_finger,num_seg,4)
        for i in range(1,num_segments):
            cummulative_quat[:,i,:] = quaternion_multiply(cummulative_quat[:,i,:],cummulative_quat[:,i-1,:])
        oriented_segments = quaternion_apply(
            cummulative_quat[:,:,None,:],   # (num_finger,num_seg,1,4)
            full_segment_pts,               # (num_finger,num_seg,N+M,3)
        )
        
        rotated_top_conn_pts = quaternion_apply(
            cummulative_quat,               # (num_finger,num_seg,4)
            top_conn_pts,                   # (num_finger,num_seg,3)
        )
        translated_rotated_top_conn_pts = rotated_top_conn_pts.clone() # (num_finger,num_seg,3)
        for i in range(1,num_segments):
            translated_rotated_top_conn_pts[:,i,:] += translated_rotated_top_conn_pts[:,i-1,:]
        rolled_offset = torch.roll(translated_rotated_top_conn_pts,shifts=1,dims=1)
        rolled_offset[:,0,:] = torch.zeros_like(rolled_offset[:,0,:]) # (num_finger,num_seg,3)
        
        return oriented_segments + rolled_offset[:,:,None,:] # (num_finger,num_seg,N+M,3)
    # endregion
    
        