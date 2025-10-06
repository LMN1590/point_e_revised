import open3d as o3d
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

cylinder_color = np.array([0.,0.,0.])
base_length = 0.1
radius = 0.02

spline_range = [0.25,4.0]
lengthen_range = [0.1,2.0]

# Control points for the segment
torch.manual_seed(85)
ctrl_tensor = torch.tensor([[0.5,0.3,0.3,0.5,0.3,0.,0.,0.,0.,0.,0.]])
ctrl_tensor.requires_grad_(True)

# Path to your .pcd file
complete_segment_path = "asset/segment/complete_segment.pcd"
base_pcd = o3d.io.read_point_cloud(complete_segment_path)
base_pts = np.asarray(base_pcd.points)
base_colors = np.asarray(base_pcd.colors)

cylinder_mask = np.all(base_colors==cylinder_color,axis=1)
cylinder_pts = torch.from_numpy(base_pts[cylinder_mask])
conn_ends = torch.from_numpy(base_pts[~cylinder_mask])

def splining(ctrl_tensor, cylinder_pts:torch.Tensor)->torch.Tensor:
    """
    Apply spline deformation to the cylinder points based on control points.
    Args:
        ctrl_tensor (torch.Tensor): Control points tensor of shape (B, 11).
        cylinder_pts (torch.Tensor): Cylinder points tensor of shape (N, 3).
    Returns:
        torch.Tensor: Modified cylinder points after spline deformation of shape (B, N, 3)
    """
    t_sample = cylinder_pts[:,1] + (base_length/2)
    
    spline_ctrl_pts = (ctrl_tensor[:,1:5])*(spline_range[1]-spline_range[0]) + spline_range[0] # (B,4)
    first_col = torch.ones(spline_ctrl_pts.shape[0],1)
    last_col = torch.ones(spline_ctrl_pts.shape[0],1)
    full_ctrl_pts = torch.cat([
        first_col,                  # (B,1)
        spline_ctrl_pts,            # (B,4)
        last_col                    # (B,1)
    ],dim=1)
    print(full_ctrl_pts)
    
    t = torch.linspace(0,base_length,6)
    coeffs = natural_cubic_spline_coeffs(t, full_ctrl_pts.T) # [6,(6,B)]
    spline = NaturalCubicSpline(coeffs)
    spline_res = spline.evaluate(t_sample) # (N,B)
    spline_res = spline_res.T.unsqueeze(-1) # (B,N,1)
    
    mod_cylinder_pts = cylinder_pts.repeat(spline_res.shape[0],1,1) # (B,N,3)
    spline_scale = torch.cat([
        spline_res,
        torch.ones_like(spline_res),
        spline_res
    ],dim=-1) # (B,N,3)
    return mod_cylinder_pts * spline_scale # (B,N,3)

def lengthening(ctrl_tensor, batched_cylinder_pts:torch.Tensor,conn_ends:torch.Tensor)->torch.Tensor:
    """
    Apply lengthening deformation to the cylinder points
    Args:
        ctrl_tensor (torch.Tensor): Control points tensor of shape (B, 11).
        batched_cylinder_pts (torch.Tensor): Cylinder points tensor of shape (B, N, 3).
        conn_ends (torch.Tensor): Connection end points tensor of shape (M, 3).
    Returns:
        torch.Tensor: Modified full segment points after lengthening deformation of shape (B, N+M, 3)
    """
    lengthen_val = ctrl_tensor[:,0] * (lengthen_range[1]-lengthen_range[0]) + lengthen_range[0] # (B,)
    print(lengthen_val)
    lengthen_val_reshaped = lengthen_val.repeat(1, batched_cylinder_pts.shape[1]).unsqueeze(-1) # (B,N,1)
    lengthen_scale = torch.cat([
        torch.ones_like(lengthen_val_reshaped),
        lengthen_val_reshaped,
        torch.ones_like(lengthen_val_reshaped)
    ]) # (B,N,3)
    
    batched_mod_cylinder_pts = batched_cylinder_pts * lengthen_scale # (B,N,3)
    return batched_mod_cylinder_pts
    
    

mod_cylinder_pts = splining(ctrl_tensor, cylinder_pts)
transformed_segments = lengthening(ctrl_tensor, mod_cylinder_pts, conn_ends)

arr = torch.cat([transformed_segments[0],conn_ends],dim=0).detach().numpy()
full_segment = o3d.geometry.PointCloud()
full_segment.points = o3d.utility.Vector3dVector(arr)
colors = np.vstack([
    np.zeros((transformed_segments[0].shape[0], 3)),
    np.full((conn_ends.shape[0], 3), 0.75)
])
full_segment.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([full_segment])
o3d.io.write_point_cloud('complete_segment.pcd', full_segment)

# t = torch.linspace(0,0.1,6)
# x = torch.tensor([0,4,16,36,64,100]).float().unsqueeze(1)
# x.requires_grad_(True)
# coeffs = natural_cubic_spline_coeffs(t, x)
# spline = NaturalCubicSpline(coeffs)

# point = torch.tensor(0.01)
# out = spline.evaluate(point)
# print(out)
    
    