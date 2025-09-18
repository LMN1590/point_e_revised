import open3d as o3d
import numpy as np
import torch
from typing import Union

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

def visualize_pcd(pcd_path: str):
    """
    Read and visualize a point cloud file (.pcd).
    
    Args:
        pcd_path (str): Path to the .pcd file
    """
    # Load
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts=np.array(pcd.points)
    print(pts.max(0),pts.min(0))

    if not pcd.has_points():
        print(f"[!] No points found in {pcd_path}")
        return

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="PCD Viewer",
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    # visualize_pcd("diff_conditioning/simulation_env/asset/fixed_base_big.pcd")
    visualize_pcd("logs/debug/softzoo/design/geometry_Batch_0_Sampling_0000_Local_0000.pcd")
    pcd = o3d.io.read_point_cloud("logs/debug/softzoo/design/geometry_Batch_0_Sampling_0000_Local_0000.pcd")
    base_points=torch.from_numpy(np.array(pcd.points))
    print(base_points.max(0).values,base_points.min(0).values)
    calibrated_base_pts = calibrate_translate_pts(
        base_points, 
        mean = torch.tensor([0.,0.,0.]), 
        scale=torch.tensor([1.,1.,1.])
    )
    print(calibrated_base_pts.max(0).values,calibrated_base_pts.min(0).values)
