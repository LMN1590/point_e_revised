import pandas as pd
import trimesh
import torch
from pytorch3d.transforms import quaternion_to_matrix
import numpy as np
from pathlib import Path
import open3d as o3d
from tqdm import tqdm

# === CONFIG ===
csv_path = "/media/aioz-nghiale/data1/Proj/point_e_revised/logs/experiment_results_4.csv"  # Input CSV
root_dir = Path("/media/aioz-nghiale/data1/Data/mujoco_scanned_objects/models")  # Mesh root directory
cube_size = 0.04  # Configurable cube size

output_csv = csv_path.replace(".csv", "_with_iou.csv")
results = []

# === LOAD CSV ===
df = pd.read_csv(csv_path)

# === LOOP OVER EACH OBJECT ===
for _, row in tqdm(df.iterrows()):
    obj_name = row["object"]
    quat = torch.tensor([row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]], dtype=torch.float32)
    scale = float(row["scale"])

    mesh_path = root_dir / obj_name / "model.obj"
    if not mesh_path.exists():
        print(f"❌ Missing mesh for {obj_name} at {mesh_path}")
        continue

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')

    # Step 1: Center mesh before transform
    center = mesh.centroid
    mesh.apply_translation(-center)

    # Step 2: Apply quaternion rotation
    quat = quat / quat.norm()
    rot_matrix = quaternion_to_matrix(quat.unsqueeze(0))[0].numpy()
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    mesh.apply_transform(transform)

    # Step 3: Apply scale
    mesh.apply_scale(scale)

    # Step 4: Recenter final AABB at origin
    bbox_min, bbox_max = mesh.bounds
    mesh.apply_translation(-(bbox_min + bbox_max) / 2)
    bbox_min, bbox_max = mesh.bounds
    aabb_extent = bbox_max - bbox_min  # [x_len, y_len, z_len]
    max_dim = np.max(aabb_extent)
    min_dim = np.min(aabb_extent)
    aspect_ratio = max_dim / (min_dim + 1e-8)  # avoid division by zero

    # === Compute IoU with cube ===
    mesh_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)

    half = cube_size / 2.0
    cube_min = np.array([-half, -half, -half])
    cube_max = np.array([half, half, half])
    cube_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=cube_min, max_bound=cube_max)

    # Manual intersection computation (Open3D CUDA-safe)
    bbox1_min = np.array(mesh_aabb.get_min_bound())
    bbox1_max = np.array(mesh_aabb.get_max_bound())
    bbox2_min = np.array(cube_aabb.get_min_bound())
    bbox2_max = np.array(cube_aabb.get_max_bound())

    inter_min = np.maximum(bbox1_min, bbox2_min)
    inter_max = np.minimum(bbox1_max, bbox2_max)
    inter_dim = np.maximum(inter_max - inter_min, 0.0)

    inter_vol = np.prod(inter_dim)
    mesh_vol = np.prod(bbox1_max - bbox1_min)
    cube_vol = np.prod(bbox2_max - bbox2_min)
    union_vol = mesh_vol + cube_vol - inter_vol
    
    outer_mesh_iou = (mesh_vol - inter_vol) / union_vol if union_vol > 0 else 0.0
    outer_cube_iou = (cube_vol - inter_vol) / union_vol if union_vol > 0 else 0.0
    
    outer_vol = (mesh_vol-inter_vol) + (cube_vol-inter_vol)*0.1
    iou = outer_vol/ union_vol if union_vol > 0 else 0.0
    iou_aspect = iou * aspect_ratio

    results.append({
        "object": obj_name,
        "quat_w": row["quat_w"],
        "quat_x": row["quat_x"],
        "quat_y": row["quat_y"],
        "quat_z": row["quat_z"],
        "scale": scale,
        "reward": row["reward"],
        "outer_mesh_iou": outer_mesh_iou,
        "outer_cube_iou": outer_cube_iou,
        "aspect_ratio": aspect_ratio,
    })
    # print(f"{obj_name}: IoU = {iou:.4f}")

    # Optional visualization (disable for batch processing)
    # scene = trimesh.Scene(mesh)
    # scene.show()

# === SAVE RESULTS ===
out_df = pd.DataFrame(results)
out_df.to_csv(output_csv, index=False)
print(f"\n✅ IoU results saved to: {output_csv}")
