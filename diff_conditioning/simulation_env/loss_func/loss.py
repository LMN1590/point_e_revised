import taichi as ti

from softzoo.utils.const import F_DTYPE,I_DTYPE

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class Loss:
    def __init__(self,parent:'LossSet',env:'BaseEnv', **kwargs):
        self.parent = parent # NOTE: check self.parent.shared_data to avoid recomputation
        self.env = env
        
        self.data = dict()
        
    def reset(self, **kwargs):
        for v in self.data.values():
            v.fill(0.)
            v.grad.fill(0.)

    def compute_final_step_loss(self, s):
        pass

    def compute_final_step_grad(self, s):
        pass

    def compute_per_step_loss(self, s):
        pass

    def compute_per_step_grad(self, s):
        pass

    def get_s_local(self, s):
        return self.env.sim.solver.get_cyclic_s(s)

    def get_loss_stats(self): # used for logging
        stats = dict()
        stats['loss'] = self.data['loss'].to_numpy().mean()
        return stats
    
    # region Helper Function
    @ti.kernel
    def _compute_n_robot_particles(self):
        # Require self.data['n_robot_particles']
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            self.data['n_robot_particles'][None] += ti.cast(is_robot, F_DTYPE)
            
    @ti.kernel
    def _compute_x_avg(self,s:I_DTYPE,s_local:I_DTYPE):
        # Require self.data['x_avg']
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                self.data['x_avg'][s] += self.env.sim.solver.x[s_local,p] / self.data['n_robot_particles'][None]

    @ti.kernel
    def _compute_v_avg_simple(self, s: I_DTYPE, s_local: I_DTYPE):
        # Require self.data['v_avg'] NOTE: no projection on orientation
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                self.data['v_avg'][s] += self.env.sim.solver.v[s_local, p] / self.data['n_robot_particles'][None]
    
    @ti.ad.grad_replaced
    def _compute_v_avg(self, s, s_local):
        self.env.design_space.compute_orientation_kernel(
            s_local,
            self.env.design_space.orientation_data['min_p'],
            self.env.design_space.orientation_data['max_p']
        )
        self._compute_v_avg_kernel(s, s_local)

    @ti.ad.grad_for(_compute_v_avg)
    def _compute_v_avg_grad(self, s, s_local):
        self.env.design_space.reset_orientation_grad() # NOTE: reset gradient every time since we use shared orientation data
        self._compute_v_avg_kernel.grad(s, s_local)
        self.env.design_space.compute_orientation_kernel.grad(
            s_local,
            self.env.design_space.orientation_data['min_p'],
            self.env.design_space.orientation_data['max_p']
        )

    @ti.kernel
    def _compute_v_avg_kernel(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                orientation = self.env.design_space.orientation_data['orientation'][None]
                v = self.env.sim.solver.v[s_local, p]
                proj_v = v.dot(orientation)
                self.data['v_avg'][s] += proj_v * orientation / self.data['n_robot_particles'][None]
    # endregion