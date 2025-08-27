import numpy as np
import open3d as o3d
import torch

# Load your point cloud
loaded_pcd = np.load('sample_generated_pcd.npz')
coords = loaded_pcd['coords']  # shape should be (N, 3)

# Convert to torch if needed
x = torch.from_numpy(coords).detach()  # you can skip .to(device) since we just save

# Convert back to numpy (just to be sure, CPU)
points_np = x.cpu().numpy()

# Create open3d point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_np)

# Optionally add colors (otherwise it's just white)
pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (points_np.shape[0], 1)))  

# Save to PLY
o3d.io.write_point_cloud("sample_generated_pcd.ply", pcd)

print("Saved point cloud to sample_generated_pcd.ply")
