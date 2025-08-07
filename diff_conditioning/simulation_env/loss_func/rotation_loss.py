import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

from typing import TYPE_CHECKING,List,Literal
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class RotationLoss(Loss):
    '''
    The loss force the robot to rotate around a certain up_direction, like if it is [1,0,0], it is suppose to rotate around the x-axis.
    '''
    def __init__(self, parent:'LossSet', env:'BaseEnv', up_direction:List[float]):
        super().__init__(parent, env)

        self.up_direction = up_direction

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            x_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
        )

        self.env.objective.draw_x = True

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_x_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)
        self._compute_x_avg.grad(s, s_local)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                up_direction = ti.Vector(self.up_direction, F_DTYPE)
                x_centered = self.env.sim.solver.x[s_local, p] - self.data['x_avg'][s]
                v_tan_dir = up_direction.cross(x_centered).normalized(NORM_EPS)
                p_loss = -v_tan_dir.dot(self.env.sim.solver.v[s_local, p])
                self.data['loss'][s] += p_loss / self.data['n_robot_particles'][None]