import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

from typing import TYPE_CHECKING,List,Literal
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class VelocityFollowingLoss(Loss):
    def __init__(self, parent:'LossSet', env:'BaseEnv', v_mul:List[float], mode:Literal[0,1]):
        '''
        Given a starting velocity and an ending velocity, the objective can generate step-by-step velocity and helps force the robot into those velocities.
        
        v_mul: A weighting vector for each velocity axis (x, y, z) — helps emphasize or ignore certain directions (e.g., [1, 0, 0] for x-axis only).

        mode:
        0: Use squared difference of full velocity vector.
        1: Split velocity into magnitude and direction, and penalize both.
        '''
        super().__init__(parent, env)

        self.v_mul = v_mul
        self.mode = mode

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            v_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
        )
        if self.mode == 1:
            self.data['loss_norm'] = ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True)
            self.data['loss_dir'] = ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True)
        self.non_learnable_data = dict(
            v_tgt=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=())
        )

        self.v_avg_mode = 1

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['v_tgt'][None] = self.env.objective.get_v_tgt(s)
        if self.v_avg_mode == 0:
            self._compute_v_avg_simple(s, s_local)
        else:
            self._compute_v_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['v_tgt'][None] = self.env.objective.get_v_tgt(s)
        self._compute_loss.grad(s, s_local)
        if self.v_avg_mode == 0:
            self._compute_v_avg_simple.grad(s, s_local)
        else:
            self._compute_v_avg_grad(s, s_local)

    def get_loss_stats(self):
        stats = super().get_loss_stats()
        if self.mode == 1:
            stats['loss_norm'] = self.data['loss_norm'].to_numpy().mean()
            stats['loss_dir'] = self.data['loss_dir'].to_numpy().mean()
        return stats

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        v_mul = ti.Vector(self.v_mul, F_DTYPE)

        v_tgt = self.non_learnable_data['v_tgt'][None]
        v_avg = self.data['v_avg'][s]

        if ti.static(self.mode == 0): # squared difference of absolute velocity
            v_diff = (v_avg - v_tgt) * v_mul
            self.data['loss'][s] = v_diff.dot(v_diff)
        elif ti.static(self.mode == 1): # split into norm and direction
            v_tgt_norm = v_tgt.norm(NORM_EPS)
            v_avg_norm = v_avg.norm(NORM_EPS)

            v_tgt_dir = v_tgt * v_mul / (v_tgt_norm + 1e-6) # NOTE: only multiply v_mul for direction
            v_avg_dir = v_avg * v_mul / (v_avg_norm + 1e-6)

            loss_norm = ti.pow(v_tgt_norm - v_avg_norm, 2)
            self.data['loss_norm'][s] = loss_norm

            loss_dir = -v_tgt_dir.dot(v_avg_dir) # cosine distance
            self.data['loss_dir'][s] = loss_dir

            self.data['loss'][s] = loss_norm * ti.cast(self.env.objective.config['weight_norm'], F_DTYPE) + \
                                loss_dir * ti.cast(self.env.objective.config['weight_direction'], F_DTYPE)
        else:
            v_diff_pos = (v_avg - v_tgt) * v_mul
            v_diff_neg = (-v_avg - v_tgt) * v_mul
            self.data['loss'][s] = ti.min(v_diff_pos.dot(v_diff_pos), v_diff_neg.dot(v_diff_neg))
