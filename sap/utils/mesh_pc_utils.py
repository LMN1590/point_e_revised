import numpy as np
import torch
import torch.nn.functional as F

import trimesh
import open3d as o3d
from skimage import measure
from igl import adjacency_matrix, connected_components
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, rasterize_meshes
from pytorch3d.ops.marching_cubes import marching_cubes
import kaolin

from typing import Union
import logging

import torch
import torch.nn.functional as F
    
def mc_from_psr(psr_grid:torch.Tensor, pytorchify=False, real_scale=False, zero_level=0):
    '''
    Run marching cubes from PSR grid
    '''
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1] # size of psr_grid
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()

    if batch_size>1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, values = measure.marching_cubes(psr_grid_numpy[i], level=0)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis = 0)
        faces = np.stack(faces, axis = 0)
        normals = np.stack(normals, axis = 0)
    else:
        try:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy, level=zero_level)
        except:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy)
    if real_scale:
        verts = verts / (s-1) # scale to range [0, 1]
    else:
        verts = verts / s # scale to range [0, 1)

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals

def verts_on_largest_mesh(verts:Union[torch.Tensor, np.ndarray], faces:Union[torch.Tensor, np.ndarray]):
    '''
    verts: Numpy array or Torch.Tensor (N, 3)
    faces: Numpy array (N, 3)
    '''
    if torch.is_tensor(faces) and torch.is_tensor(verts):
        verts = verts.squeeze().detach().cpu().numpy()
        faces = faces.squeeze().int().detach().cpu().numpy()

    A = adjacency_matrix(faces)
    num, conn_idx, conn_size = connected_components(A)
    if num == 0:
        v_large, f_large = verts, faces
    else:
        max_idx = conn_size.argmax() # find the index of the largest component
        v_large = verts[conn_idx==max_idx] # keep points on the largest component

        if True:
            mesh_largest = trimesh.Trimesh(verts, faces)
            connected_comp = mesh_largest.split(only_watertight=False)
            mesh_largest = connected_comp[max_idx]
            v_large, f_large = mesh_largest.vertices, mesh_largest.faces
            v_large = v_large.astype(np.float32)
    return v_large, f_large

def export_pointcloud(name, points, normals=None):
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)
    

def mesh_rasterization(verts, faces, pose, img_size):
    '''
    Use PyTorch3D to rasterize the mesh given a camera 
    '''
    transformed_v = pose.transform_points(verts.detach()) # world -> ndc coordinate system
    if isinstance(pose, PerspectiveCameras):
        transformed_v[..., 2] = 1/transformed_v[..., 2]
    # find p_closest on mesh of each pixel via rasterization
    transformed_mesh = Meshes(verts=[transformed_v], faces=[faces])
    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        transformed_mesh,
        image_size=img_size,
        blur_radius=0,
        faces_per_pixel=1,
        perspective_correct=False
    )
    pix_to_face = pix_to_face.reshape(1, -1) # B x reso x reso -> B x (reso x reso)
    mask = pix_to_face.clone() != -1
    mask = mask.squeeze()
    pix_to_face = pix_to_face.squeeze()
    w = bary_coords.reshape(-1, 3)

    return pix_to_face, w, mask

def calc_inters_points(verts, faces, pose, img_size, mask_gt=None):
    verts = verts.squeeze()
    faces = faces.squeeze()
    pix_to_face, w, mask = mesh_rasterization(verts, faces, pose, img_size)
    if mask_gt is not None:
        #! only evaluate within the intersection
        mask = mask & mask_gt
    # find 3D points intesected on the mesh
    if True:
        w_masked = w[mask]
        f_p = faces[pix_to_face[mask]].long() # cooresponding faces for each pixel
        # corresponding vertices for p_closest
        v_a, v_b, v_c = verts[f_p[..., 0]], verts[f_p[..., 1]], verts[f_p[..., 2]]
        
        # calculate the intersection point of each pixel and the mesh
        p_inters = w_masked[..., 0, None] * v_a + \
                w_masked[..., 1, None] * v_b + \
                w_masked[..., 2, None] * v_c
    else:
        # backproject ndc to world coordinates using z-buffer
        W, H = img_size[1], img_size[0]
        xy = uv.to(mask.device)[mask]
        x_ndc = 1 - (2*xy[:, 0]) / (W - 1)
        y_ndc = 1 - (2*xy[:, 1]) / (H - 1)
        z = zbuf.squeeze().reshape(H * W)[mask]
        xy_depth = torch.stack((x_ndc, y_ndc, z), dim=1)
        
        p_inters = pose.unproject_points(xy_depth, world_coordinates=True)
    
        # if there are outlier points, we should remove it
        if (p_inters.max()>1) | (p_inters.min()<-1):
            mask_bound = (p_inters>=-1) & (p_inters<=1)
            mask_bound = (mask_bound.sum(dim=-1)==3)
            mask[mask==True] = mask_bound
            p_inters = p_inters[mask_bound]
            print('!!!!!find outlier!')

    return p_inters, mask, f_p, w_masked

def sample_pc_in_mesh(mesh,num_points:int = 10000, density:float = 1.0, voxel_size:float = 0.05):
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    center = aabb.get_center()
    scale = np.max(max_bound - min_bound)  # largest dimension
    mesh = mesh.translate(-center).scale(1.0/scale, center=np.zeros(3))
    
    # Convert to tensor mesh for watertight test
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    # bounding box for uniform sampling
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound

    # try watertight sampling first
    total_samples = int(num_points * density * 2)
    pts = np.random.uniform(low=min_bound, high=max_bound, size=(total_samples, 3)).astype(np.float32)
    occ = scene.compute_occupancy(o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32))
    inside_pts = pts[occ.numpy() > 0]
    
    if inside_pts.shape[0] < num_points // 2:
        logging.warning("Warning!!! Insufficient points sampled. Mesh may not be watertight → falling back to voxelization")

        # Voxelize mesh
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

        voxel_points = []
        for voxel in voxel_grid.get_voxels():
            center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            # jitter inside voxel for denser sampling
            jitter = np.random.uniform(-voxel_size/2, voxel_size/2, size=(3,))
            voxel_points.append(center + jitter)

        inside_pts = np.array(voxel_points)
    
    # Randomly pick desired number
    if inside_pts.shape[0] > num_points:
        idx = np.random.choice(inside_pts.shape[0], num_points, replace=False)
        inside_pts = inside_pts[idx]

    # Return as point cloud
    inside_pts = inside_pts * scale + center
    return inside_pts


def sample_pc_in_mesh_gpu_optim(
    v: torch.Tensor,
    f: torch.Tensor,
    num_points: int = 10000,
    density: float = 1.0,
    voxel_size: float = 0.05,
):
    """
    GPU-optimized mesh sampling using Kaolin check_sign.
    
    Args:
        v (torch.Tensor): (V, 3) vertices of the mesh
        f (torch.Tensor): (F, 3) faces (indices into v)
        num_points (int): number of points to sample inside mesh
        density (float): oversampling density factor
        voxel_size (float): approximate fallback voxel jitter size
    """
    device = v.device

    # --- Normalize mesh like in your O3D code ---
    min_bound, _ = torch.min(v, dim=0)
    max_bound, _ = torch.max(v, dim=0)
    center = torch.mean(v,dim=0)
    scale = torch.max(max_bound - min_bound)
    v_norm = (v - center) / scale
    v_norm = v_norm.unsqueeze(0)
    
    # Uniform AABB sampling
    total_samples = int(num_points * density * 2)
    pts = torch.rand((1, total_samples, 3), device=device) * 2.0 - 1.0  # [-1,1]^3 cube
    # shrink to AABB of normalized mesh
    min_bound_n, _ = torch.min(v_norm[0], dim=0)
    max_bound_n, _ = torch.max(v_norm[0], dim=0)
    pts = pts * (max_bound_n - min_bound_n) / 2.0 + (max_bound_n + min_bound_n) / 2.0
    
    # Inside-outside test
    occ = kaolin.ops.mesh.check_sign(v_norm, f.to(torch.int64), pts)  # (1, total_samples)
    inside_pts = pts[0, occ[0]]
    if inside_pts.shape[0] < num_points // 2:
        logging.warning("Warning!!! Insufficient points sampled. Mesh may not be watertight → falling back to voxelization")
        grid_coords = []
        for dim in range(3):
            axis = torch.arange(
                start = min_bound_n[dim].item(), 
                end= max_bound_n[dim].item(), 
                step = voxel_size, 
                device=device
            )
            grid_coords.append(axis)
        gx, gy, gz = torch.meshgrid(grid_coords, indexing="ij")
        grid_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        
        # Add jitter for variety
        jitter = (torch.rand_like(grid_pts) - 0.5) * (voxel_size/scale)
        grid_pts = grid_pts + jitter

        occ_vox = kaolin.ops.mesh.check_sign(v_norm, f.to(torch.int64), grid_pts.unsqueeze(0))
        inside_pts = grid_pts[occ_vox[0]]
        
    if inside_pts.shape[0] > num_points:
        idx = torch.randperm(inside_pts.shape[0], device=device)[:num_points]
        inside_pts = inside_pts[idx]
        
    inside_pts = inside_pts * scale + center
    return inside_pts