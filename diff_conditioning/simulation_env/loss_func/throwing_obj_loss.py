import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

from typing import TYPE_CHECKING,List,Literal
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class ThrowingObjectLoss(Loss):
    '''
    The loss forces the robot to interact with an object so that it ends up as far as possible in certain direction
    '''
    
    def __init__(self, parent:'LossSet', env:'BaseEnv', obj_particle_id:int=2, x_mul:List[float]=[1., 0., 0.]):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_object_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
        )
        self.obj_particle_id = obj_particle_id
        self.x_mul = x_mul

    def reset(self):
        super().reset()
        self._compute_n_object_particles()

    def compute_final_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss(s, s_local)

    def compute_final_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)

    @ti.func
    def _is_object(self, id):
        return id == self.obj_particle_id

    @ti.kernel
    def _compute_n_object_particles(self):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id):
                self.data['n_object_particles'][None] += ti.cast(1, F_DTYPE)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id) and self.env.sim.solver.p_rho[p] > 0:
                x_mul = ti.Vector(self.x_mul, F_DTYPE) / self.data['n_object_particles'][None]
                p_loss = -(ti.abs(self.env.sim.solver.x[s_local, p]-ti.Vector([0,0.2,0])) * x_mul).sum()
                self.data['loss'][s] += p_loss # TODO: compute difference from initial position