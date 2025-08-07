import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

from typing import TYPE_CHECKING,List,Literal
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class WaypointFollowingLoss(Loss):
    '''
    It is like the velocity one(given start and end position and velocity, they generate a step-by-step position and velocity checkpoint for the robot to follow)
    '''
    
    def __init__(self, parent:'LossSet', env:'BaseEnv'):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            x_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
        )
        self.non_learnable_data = dict(
            x_tgt=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=())
        )
        self.env.objective.draw_x = True

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['x_tgt'][None] = self.env.objective.get_x_tgt(s)
        self._compute_x_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['x_tgt'][None] = self.env.objective.get_x_tgt(s)
        self._compute_loss.grad(s, s_local)
        self._compute_x_avg.grad(s, s_local)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        x_tgt = self.non_learnable_data['x_tgt'][None]
        x_avg = self.data['x_avg'][s]

        x_diff = x_tgt - x_avg
        self.data['loss'][s] = x_diff.dot(x_diff)