import open3d as o3d
import numpy as np

# Path to your PCD file
pcd_path = "diff_conditioning/simulation_env/asset/complete_segment_h05_r01.pcd"

# Load the point cloud
pcd = o3d.io.read_point_cloud(pcd_path)
height = 0.5
radius = 0.1
cylinder_scale = 0.3
radius_scale = 1.

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)


cylinder_mask = np.all(colors==[0,0,0],axis=1)
cylinder_pts = points[cylinder_mask] 
conn_ends = points[~cylinder_mask]
print(cylinder_pts.max(0), cylinder_pts.min(0))
print(conn_ends.max(0), conn_ends.min(0))
print("---------------------------")

cylinder_pts = points[cylinder_mask] * radius_scale * [1.,cylinder_scale/radius_scale,1.]
conn_ends = points[~cylinder_mask] * radius_scale   
print(cylinder_pts.max(0), cylinder_pts.min(0))
print(conn_ends.max(0), conn_ends.min(0))
print("---------------------------")

diff_height = height/2 * (radius_scale-cylinder_scale)
print(diff_height)
top_conn_ends = conn_ends[conn_ends[:,1]>0] - [0.,diff_height,0.]
bottom_conn_ends = conn_ends[conn_ends[:,1]<0] + [0.,diff_height,0.]
print(cylinder_pts.max(0), cylinder_pts.min(0))
print(top_conn_ends.max(0), top_conn_ends.min(0))
print(bottom_conn_ends.max(0), bottom_conn_ends.min(0))
print("---------------------------")

cylinder_pcd = o3d.geometry.PointCloud()
cylinder_pcd.points = o3d.utility.Vector3dVector(cylinder_pts)
cylinder_pcd.paint_uniform_color([0, 0, 0])  # keep black

conn_pcd = o3d.geometry.PointCloud()
conn_pcd.points = o3d.utility.Vector3dVector(np.vstack([top_conn_ends, bottom_conn_ends]))
conn_pcd.paint_uniform_color([1, 0, 0])  # for visualization, make it red

# --- (Optional) Merge and save combined version ---
merged_pcd = cylinder_pcd + conn_pcd
o3d.io.write_point_cloud(f"complete_segment_h{str(height*cylinder_scale).replace('.','')}_r{str(radius*radius_scale).replace('.','')}.pcd", merged_pcd)

o3d.visualization.draw_geometries([merged_pcd])