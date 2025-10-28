import open3d as o3d
import numpy as np

# Path to your PCD file
pcd_path = "diff_conditioning/simulation_env/asset/complete_segment_h05_r01.pcd"

# Load the point cloud
pcd = o3d.io.read_point_cloud(pcd_path)

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Define color thresholds (for (0,0,0) cylinder)
# Small tolerance because PCD colors may not be exactly 0 due to float conversion
tolerance = 1e-3
is_cylinder = np.all(colors < tolerance, axis=1)
is_connection = ~is_cylinder  # everything else

# Separate point clouds
pcd_cylinder = pcd.select_by_index(np.where(is_cylinder)[0])
pcd_connection = pcd.select_by_index(np.where(is_connection)[0])

# Further split connections by Y coordinate
points_connection = np.asarray(pcd_connection.points)
upper_idx = np.where(points_connection[:, 1] > 0)[0]
lower_idx = np.where(points_connection[:, 1] <= 0)[0]

pcd_upper = pcd_connection.select_by_index(upper_idx)
pcd_lower = pcd_connection.select_by_index(lower_idx)

# Compute bounding boxes
bbox_cylinder = pcd_cylinder.get_axis_aligned_bounding_box()
bbox_upper = pcd_upper.get_axis_aligned_bounding_box()
bbox_lower = pcd_lower.get_axis_aligned_bounding_box()

# Color bounding boxes for visualization
bbox_cylinder.color = (0, 0, 0)      # black
bbox_upper.color = (1, 0, 0)         # red
bbox_lower.color = (0, 0, 1)         # blue

# Print bounding box coordinates
print("Cylinder bbox min:", bbox_cylinder.get_min_bound(), "max:", bbox_cylinder.get_max_bound())
print("Upper connection bbox min:", bbox_upper.get_min_bound(), "max:", bbox_upper.get_max_bound())
print("Lower connection bbox min:", bbox_lower.get_min_bound(), "max:", bbox_lower.get_max_bound())

# Visualize everything
o3d.visualization.draw_geometries(
    [pcd, bbox_cylinder, bbox_upper, bbox_lower],
    window_name="Cylinder + Connections Bounding Boxes",
    width=1000,
    height=800,
)
