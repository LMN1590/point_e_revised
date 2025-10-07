import torch
import numpy as np

import open3d as o3d

from typing import Union
import matplotlib.pyplot as plt

from softzoo.utils.general_utils import extract_part_pca_inner,row_permutation,extract_part_pca
from softzoo.utils.visualization_utils import get_arrow

def calibrate_translate_pts(
    points:torch.Tensor,mean:torch.Tensor,
    flipped_x:bool = False,flipped_y:bool = False,flipped_z:bool=False,
    scale:Union[float,torch.Tensor]=1.0
)->torch.Tensor:
    points_mean = points.mean(0)
    norm_points = points - points_mean
    norm_points = norm_points * scale
    if flipped_x: norm_points[:,0] = 0.-norm_points[:,0]
    if flipped_y: norm_points[:,1] = 0.-norm_points[:,1]
    if flipped_z: norm_points[:,2] = 0.-norm_points[:,2]
    points_calibrated = norm_points + mean
    return points_calibrated

def find_mid_lowest_pt(points: np.ndarray,w: float = 0.2,eps: float = 1e-8):
    """
    points: (N,3) numpy array
    w: weight for centrality in XZ (0..1). higher -> more central
    soft: if True, return virtual weighted average; else return existing argmin point
    temperature: soft selection temperature (lower -> more peaky)
    """
    xyz = points
    xz = xyz[:, [0, 2]]
    # centroid in xz
    center_xz = xz.mean(axis=0)
    dists_xz = np.linalg.norm(xz - center_xz[None, :], axis=1)
    y = xyz[:, 1]

    y_min, y_max = y.min(), y.max()
    d_max = dists_xz.max()

    # normalize to [0,1], smaller is better
    yhat = (y - y_min) / (y_max - y_min + eps)
    yhat = np.clip(yhat, 0.0, 1.0)
    dhat = dists_xz / (d_max + eps)
    dhat = np.clip(dhat, 0.0, 1.0)
    
    score = (1.0 - w) * yhat + w * dhat
    idx = np.argmin(score)
    return xyz[idx]

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

def rotate_y(points, angle):
    """
    Rotate 3D points around the Y-axis (X-Z plane rotation).

    Args:
        points: [N,3] numpy array of points
        angle: float, rotation angle in radians

    Returns:
        rotated: [N,3] numpy array of rotated points
    """
    rot_mat = torch.tensor([
        [torch.cos(angle), 0, torch.sin(angle)],
        [0, 1, 0],
        [-torch.sin(angle), 0, torch.cos(angle)]
    ])
    return points @ rot_mat.T  # [N,3] x [3,3]
# endregion