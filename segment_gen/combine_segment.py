import open3d as o3d
import matplotlib.pyplot as plt

import numpy as np

# Path to your .pcd file
core_path = "asset/segment/core_segment_longer.pcd"
conn_end_path = "asset/segment/conn_end.pcd"

# Read the point cloud
core_pcd = o3d.io.read_point_cloud(core_path)
core_arr = np.asarray(core_pcd.points)
top_core = core_arr[core_arr[:,1]>=2.5]
bottom_core = core_arr[core_arr[:,1]<=-2.5]
rest_core = core_arr[(core_arr[:,1]<2.5) & (core_arr[:,1]>-2.5)]

conn_end_pcd = o3d.io.read_point_cloud(conn_end_path)
conn_end_arr = np.asarray(conn_end_pcd.points)
bottom_arr = conn_end_arr[conn_end_arr[:,1]<0]
bottom_arr[:,1] -= 3.5
top_arr = conn_end_arr[conn_end_arr[:,1]>=0]
top_arr[:,1] += 3.5

colors = np.vstack([
    np.full((top_arr.shape[0], 3), 0.75),
    np.full((top_core.shape[0], 3), 0.75),
    np.zeros((rest_core.shape[0], 3)),
    np.full((bottom_core.shape[0], 3), 0.75),
    np.full((bottom_arr.shape[0], 3), 0.75),
])


arr = np.concatenate([top_arr,top_core,rest_core,bottom_core,bottom_arr],axis=0)*0.1
full_segment = o3d.geometry.PointCloud()
full_segment.points = o3d.utility.Vector3dVector(arr)
full_segment.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([full_segment])
o3d.io.write_point_cloud('complete_segment.pcd', full_segment)
print(arr.max(axis=0),arr.min(axis=0))
print(arr.shape)