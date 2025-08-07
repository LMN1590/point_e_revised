import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE

from typing import TYPE_CHECKING,List
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv
    
@ti.data_oriented
class TrajectoryFollowingLoss(Loss):
    '''
    The loss helps the robot reaches a stated goal at the specific timestep, by showing them a trajectory to follow
    '''
    def __init__(self, parent:'LossSet', env:'BaseEnv', goal:List[float]):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            x_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(env.sim.solver.max_substeps), needs_grad=True),
            traj=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(env.sim.solver.max_substeps), needs_grad=True),
        )
        self.goal = goal # (x, y, z, s)

    def reset(self):
        super().reset()
        s = 0
        s_local = self.get_s_local(s)
        self._compute_n_robot_particles()
        self._compute_x_avg(s, s_local) # get robot initial position as reference
        self._prepare_traj(s0=s)

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_x_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)
        self._compute_x_avg.grad(s, s_local)

    def _prepare_traj(self, s0):
        self._terrain_info = [v for v in self.env.sim.solver.static_component_info if v['type'] == 'Static.Terrain'][0]
        self._set_traj(s0)

    @ti.kernel
    def _set_traj(self, s0: I_DTYPE):
        max_s = int(self.goal[3])
        for s in range(max_s):
            # get waypoints
            goal_xyz = ti.Vector(self.goal[:3], F_DTYPE)
            traj_xyz_at_s = goal_xyz * s / ti.cast(self.goal[3], F_DTYPE) + self.data['x_avg'][s0]
            
            # conversion from mpm resolution to terrain surface resolution
            base = traj_xyz_at_s * self.env.sim.solver.inv_dx - 0.5
            padding = ti.cast(self.env.sim.solver.padding, F_DTYPE)
            base = (base - padding) / (ti.cast(self.env.sim.solver.n_grid, F_DTYPE) - 2 * padding)
            I = ti.cast(base * self._terrain_info['resolution'], I_DTYPE)
            i = ti.min(ti.max(I[0], 0), self._terrain_info['resolution'] - 1)
            j = ti.min(ti.max(I[2], 0), self._terrain_info['resolution'] - 1)

            # set trajectory
            surface_point = self._terrain_info['polysurface_points'][i, j]
            # surface_normal = self._terrain_info['polysurface_normals'][i, j] # NOTE: use normal may lead to self-intersecting trajectory

            height = ti.cast(self.goal[1], F_DTYPE)
            self.data['traj'][s] = surface_point + height

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        self.data['loss'][s] = (self.data['x_avg'][s] - self.data['traj'][s]).norm()