import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

import trimesh
import open3d as o3d
import skimage

import enum
import os
from plyfile import PlyData

class Enum(enum.Enum):
    @classmethod
    def members(cls):
        return cls.__members__.values()

    @classmethod
    def is_member(cls, inp):
        return (inp in cls.__members__) or (inp.name in cls.__members__)

def extract_part_pca(pcd, return_part_colors=False, within_part_clustering=True):
    '''
    Returns
    -------
    all_part_pca_components : List of pca components of clusters
    all_part_pca_singular_values : The singular values after performing PCA
    all_part_pc : List of points inside of a cluster
    unique_colors: List of unique colors
    """
    '''
    
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    
    unique_colors = np.unique(colors, axis=0)
    
    disc_colors = np.array(plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors)
    if np.any(unique_colors == np.array([0., 0., 0.])):
        disc_colors = np.concatenate([
            np.array([[0., 0., 0.]]), # gray
            disc_colors
        ],axis=0)
    
    unique_colors = disc_colors[:len(unique_colors),:3]
    return extract_part_pca_inner(points,colors,return_part_colors,within_part_clustering,unique_colors)
    
    
    
def extract_part_pca_inner(points,lbls,return_part_colors=False,within_part_clustering=True, unique_lbls = []): 
    points_std = StandardScaler().fit_transform(points)
    
    all_part_pc = dict()
    all_part_pc_pca = dict()
    all_part_pc_std = dict()
    
    for i, unique_color in enumerate(unique_lbls):
        mask = np.all(lbls == unique_color[None,:], axis=1)
        masked_points = points[mask]
        masked_points_std = points_std[mask]
        
        if within_part_clustering: # HACK
            within_part_pcd = o3d.geometry.PointCloud()
            within_part_pcd.points = o3d.utility.Vector3dVector(masked_points)
            
            eps = max(
                masked_points.max(0) - masked_points.min(0)
            ) / np.power(masked_points.shape[0], 1/3)
            
            within_part_labels = np.array(within_part_pcd.cluster_dbscan(eps=eps, min_points=10))
            unique_within_part_labels = np.unique(within_part_labels)
            n_within_part_labels = len(unique_within_part_labels)
            
            if -1 in unique_within_part_labels:
                n_within_part_labels -= 1
                
            if n_within_part_labels > 1:
                within_part_mask = within_part_labels == 0
                all_part_pc_pca[i] = masked_points[within_part_mask]
                all_part_pc_std[i] = masked_points_std[within_part_mask]
            else:
                all_part_pc_pca[i] = masked_points
                all_part_pc_std[i] = masked_points_std
            all_part_pc[i] = masked_points
        else:
            all_part_pc[i] = masked_points
            all_part_pc_pca[i] = masked_points
            all_part_pc_std[i] = masked_points_std

    all_part_pca_components = dict()
    all_part_pca_singular_values = dict()
    for k, part_pc_pca in all_part_pc_pca.items():
        pca = PCA(n_components=3)
        pca.fit(part_pc_pca)
        all_part_pca_components[k] = pca.components_
        all_part_pca_singular_values[k] = pca.singular_values_

    if return_part_colors:
        return all_part_pca_components, all_part_pca_singular_values, all_part_pc, unique_lbls
    else:
        return all_part_pca_components, all_part_pca_singular_values, all_part_pc   

def recursive_getattr(obj, name):
    out = obj
    for v in name.split('.'):
        assert hasattr(out, v), f'{out} has no attribute {v}'
        out = getattr(out, v)
        
    return out

def load_mesh(fn, scale=1, offset=(0, 0, 0)):
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    print(f'loading {fn}')
    if os.path.splitext(fn) == '.ply':
        plydata = PlyData.read(fn)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        elements = plydata['face']
        vertex_indices = elements['vertex_indices']
    else:
        mesh = trimesh.load(fn, skip_materials=True)
        vertices = np.array(mesh.vertices)
        vertex_indices = np.array(mesh.faces)
        x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
    num_tris = len(vertex_indices)
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(vertex_indices):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale[0] + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale[1] + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale[2] + offset[2]

    return triangles

def load_points_from_mesh(fn, scale=1, offset=(0, 0, 0), num_points=5000):
    mesh = o3d.io.read_triangle_mesh(fn)
    pc = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pc.points)
    points = points * np.array(scale) + np.array(offset)

    return points

def pcd_to_mesh(pcd, strategy='voxel', voxel_size=None):
    """ Convert Open3d PCD object to Open3d Mesh object. """
    if strategy == 'bpa':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        
        mesh = mesh.simplify_quadric_decimation(100000)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
    elif strategy == 'poisson':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
    elif strategy == 'voxel':
        # convert point cloud to voxel grid
        if voxel_size is None:
            pts = np.asarray(pcd.points)
            voxel_size = min(pts.max(0) - pts.min(0)) / 10 # 5
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        # convert voxel grid indices to voxel grid (fake) sdf
        padding = 3
        voxels = voxel_grid.get_voxels()
        mesh = o3d.geometry.TriangleMesh()
        if len(voxels) > 0:
            indices = np.stack(list(vx.grid_index for vx in voxels))
            indices_shape = indices.max(0) + 1
            voxel_grid_shape = indices_shape + 2 * padding
            voxels_np = np.ones(voxel_grid_shape)
            for idx in indices:
                idx += padding
                voxels_np[idx[0], idx[1], idx[2]] = -1.

            # convert voxel sdf to mesh (in voxel grid coordinate)
            verts, faces, normals, values = skimage.measure.marching_cubes(voxels_np)
            
            # normalize to original coordinate
            voxel_grid_extents = voxel_grid.get_max_bound() - voxel_grid.get_min_bound()
            verts = (verts - padding) / indices_shape * voxel_grid_extents + voxel_grid.get_min_bound()

            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
    else:
        raise ValueError(f'Unrecognized strategy {strategy} to convert point cloud to mesh')

    return mesh


def save_pcd_to_mesh(filepath, pcd):
    mesh = pcd_to_mesh(pcd)
    o3d.io.write_triangle_mesh(filepath, mesh)
    
def surface_to_mesh(surface: np.ndarray):
    vertices = []
    indices = []
    grid_size = surface.shape
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            vertices.append(surface[i, j])
            if i < (grid_size[0] - 1) and j < (grid_size[1] - 1):
                index = i * grid_size[0] + j
                a = index
                b = index + 1
                c = index + grid_size[0] + 1
                d = index + grid_size[0]
                indices.append((a, b, c))
                indices.append((a, c, d))
    vertices, indices = np.array(vertices), np.array(indices)

    return vertices, indices

def cartesian_np(coord_list):
    idcs_list = list(range(len(coord_list) + 1))
    idcs_list = [idcs_list[1], idcs_list[0]] + idcs_list[2:]
    out = np.stack(np.meshgrid(*coord_list), -1).transpose(*idcs_list)
    return out

def row_permutation(A: torch.Tensor, B: torch.Tensor) -> torch.LongTensor:
    """
    Given A, B of shape (N,3) containing the same rows in different order,
    return a LongTensor perm of length N so that
        B[i] == A[ perm[i] ]
    """
    # 1) unique on A
    _, invA = torch.unique(A,   dim=0, return_inverse=True)
    # 2) unique on B
    _, invB = torch.unique(B,   dim=0, return_inverse=True)
    # 3) invert invA
    positions = torch.empty_like(invA)
    positions[invA] = torch.arange(A.size(0), device=A.device)
    # 4) map
    return positions[invB]

if __name__ == "__main__":
    # Example usage
    pcd = o3d.io.read_point_cloud("logs/artifacts/grippers/0/fingerl_mesh.pcd")  # Replace with your point cloud file
    components, singular_values, part_pc, colors = extract_part_pca(pcd, return_part_colors=True)
    print("PCA Components:", components[0].shape)
    print("Singular Values:", singular_values[0].shape)
    print("Part Point Clouds:", part_pc[0].shape)
    print("Colors:", colors[0].shape)