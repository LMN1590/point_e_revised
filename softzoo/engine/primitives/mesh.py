import numpy as np
import taichi as ti
import math
import time
from pyquaternion import Quaternion

from typing import TYPE_CHECKING,Union

from .primitive_base import PrimitiveBase
from ...utils.const import I_DTYPE, F_DTYPE, NORM_EPS
from ...utils.general_utils import load_mesh
from ...utils.voxelizer import Voxelizer
from ..materials import Material
from ...configs.config_dataclass import MeshConfig

if TYPE_CHECKING:
    from ..mpm_solver import MPMSolver

def apply_rotation_to_triangles(triangles, rotation_matrix, center):
    """
    Apply rotation to triangles around a center point
    """
    # Reshape triangles for easier manipulation
    vertices = triangles.reshape(-1, 3)
    
    # Translate to origin (relative to center)
    vertices_centered = vertices - center
    
    # Apply rotation
    vertices_rotated = np.dot(vertices_centered, rotation_matrix.T)
    
    # Translate back
    vertices_final = vertices_rotated + center
    
    # Reshape back to triangle format
    return vertices_final.reshape(triangles.shape)


@ti.data_oriented
class Mesh(PrimitiveBase):
    def __init__(self, solver:'MPMSolver', cfg:MeshConfig):
        super().__init__(solver, cfg)
        assert self.solver.dim == 3

        self.grid_size = self.solver.n_grid # NOTE: grid size is a different thing in taichi_element implementation

        self.voxelizer = Voxelizer(
            res=self.solver.res,
            dx=self.solver.dx,
            precision=self.solver.f_dtype,
            padding=self.solver.padding,
            super_sample=self.voxelizer_super_sample
        )
        
        # Instantiate shape in MPM
        if not self.is_rigid:
            triangles = load_mesh(self.file_path, self.scale, offset=(0, 0, 0))  # Step 1

            # Step 2
            bbox_min = triangles[:, 0::3].min(axis=(0, 1))
            bbox_max = triangles[:, 0::3].max(axis=(0, 1))
            bbox_center = (bbox_min + bbox_max) / 2

            # Step 3
            target_center = np.array(self.initial_position)
            offset = target_center - bbox_center
            triangles += np.tile(offset, 3)
            
            # Step 3.5: Perform Rotation
            rot_matrix = Quaternion(*self.initial_rotation).rotation_matrix # uses w,x,y,z
            triangles  = apply_rotation_to_triangles(triangles,rot_matrix,target_center)

            # Step 4
            new_bbox_min_y = triangles[:, 1::3].min()
            min_allowed_y = cfg.ground_height #0.1
            if new_bbox_min_y < min_allowed_y:
                lift = min_allowed_y - new_bbox_min_y
                triangles[:, 1::3] += lift

            # Continue as before
            self.voxelizer.voxelize(triangles)
            self.source_bound = ti.Vector.field(self.solver.dim, dtype=F_DTYPE, shape=1)
            for i in range(self.solver.dim):
                self.source_bound[0][i] = 0.0  # No additional offset needed

            self.seed(self.solver.current_s, -1, self.material.value, self.particle_id)
            ti.sync() # NOTE: not sure why we need this; it seems like it's ok without this     

    @ti.kernel
    def seed(self, s: I_DTYPE, num_new_particles: I_DTYPE, material: I_DTYPE, particle_id: I_DTYPE):
        for i, j, k in self.voxelizer.voxels:
            inside = 1
            if ti.static(False): # NOTE: not working
                for d in ti.static(range(3)):
                    inside = inside and -self.grid_size // 2 + self.solver.padding <= i \
                                and i < self.grid_size // 2 - self.solver.padding

            if inside and self.voxelizer.voxels[i, j, k] > 0:
                for l in range(self.sample_density + 1):
                    ss = self.sample_density / self.voxelizer_super_sample**self.solver.dim
                    if ti.random() + l < ss:
                        x = ti.Vector([
                            ti.random() + i,
                            ti.random() + j,
                            ti.random() + k
                        ], dt=F_DTYPE) * (
                            self.solver.dx / self.voxelizer_super_sample
                        ) + self.source_bound[0]

                        v = ti.Vector(self.initial_velocity, F_DTYPE)
                        self.solver.seed_nonoverlap_particle(s, x, v, material, particle_id, self.particle_info.p_rho_0,
                                                             self.particle_info.mu_0, self.particle_info.lambd_0)
