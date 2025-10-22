import open3d as o3d

# === Load mesh (.obj, .ply, .stl, etc.) ===
mesh = o3d.io.read_triangle_mesh("asset/obj_placeholder/bottle.obj")

# Optional: check if normals are present, otherwise compute
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# === Sample points from the mesh surface ===
# You can adjust the number of points for desired density
pcd = mesh.sample_points_uniformly(number_of_points=2000)

# Optional: compute normals for the point cloud
# pcd.estimate_normals()

o3d.visualization.draw_geometries(
    [pcd],
    window_name="PointCloud Viewer",
    width=800,
    height=600,
    point_show_normal=False
)