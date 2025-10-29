import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

from typing import TYPE_CHECKING,List,Literal
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv
    
@ti.data_oriented
class ContactLoss(Loss):
    '''
    The Contact Loss that enforce the proximity force between the finger's surface and the object surface by applying pulling forces towards each other when they are close enough, below a certain threshold.
    '''
    def __init__(
        self,parent:'LossSet', env:'BaseEnv',
        obj_particle_id:int
    ):
        super().__init__(parent, env)
        self.obj_particle_id = obj_particle_id
        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles = ti.field(
                dtype=F_DTYPE,
                shape=(),
                needs_grad=True
            )
        )

    def reset(self):
        super().reset()
    
    def compute_final_step_loss(self, s):
        self._compute_n_robot_particles()
        s_local = self.get_s_local(s)
        self._compute_loss(s, s_local)

    def compute_final_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)
        
    @ti.func
    def _is_object(self,id):
        return id == self.obj_particle_id
    
    @ti.kernel
    def _compute_loss(self, s:I_DTYPE, s_local:I_DTYPE):
        total_dist = 0.0
        n_particles = self.env.sim.solver.n_particles[None]
        
        for p_robot in range(n_particles):
            id_robot = self.env.sim.solver.particle_ids[p_robot]
            is_robot = self.env.design_space.is_robot(id_robot) and (self.env.sim.solver.p_rho[p_robot] > 0)
            if is_robot:
                min_dist = ti.cast(1e5,F_DTYPE)
                pos_r = self.env.sim.solver.x[s_local, p_robot]
                for p_obj in range(n_particles):
                    id_obj = self.env.sim.solver.particle_ids[p_obj]
                    is_object = self._is_object(id_obj) and (self.env.sim.solver.p_rho[p_obj] > 0)
                    if is_object:
                        pos_o = self.env.sim.solver.x[s_local, p_obj]
                        dist = (pos_r - pos_o).norm()
                        min_dist = ti.min(min_dist, dist)
                total_dist += min_dist / (self.data['n_robot_particles'])

        self.data['loss'][s] = total_dist
        
        