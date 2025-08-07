import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE

from typing import TYPE_CHECKING,List
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class FinalStepCoMLoss(Loss):
    '''
    The loss is for helping the robot reaches the furthest distance possible in the direction stated by x_mul
    '''
    def __init__(self,parent:'LossSet',env:'BaseEnv',x_mul:List[float] = [1.,0.,0.]):
        super().__init__(parent,env)
        
        self.data = dict(
            loss = ti.field(
                dtype = F_DTYPE,
                shape = (self.env.sim.solver.max_substeps),
                needs_grad=True
            ),
            n_robot_particles = ti.field(
                dtype=F_DTYPE,
                shape=(),
                needs_grad=True
            )
        )
        
        self.x_mul = x_mul
        
    def reset(self):
        super().reset()
        
    def compute_final_step_loss(self, s):
        self._compute_n_robot_particles()
        s_local = self.get_s_local(s)
        self._compute_loss(s, s_local)

    def compute_final_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                x_mul = ti.Vector(self.x_mul, F_DTYPE) / self.data['n_robot_particles'][None]
                p_loss = -(self.env.sim.solver.x[s_local, p] * x_mul).sum()
                self.data['loss'][s] += p_loss # TODO: compute difference from initial position