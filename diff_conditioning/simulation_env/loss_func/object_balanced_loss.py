import taichi as ti

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

from typing import TYPE_CHECKING,List,Literal
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv
    
@ti.data_oriented
class ObjectBalancedLoss(Loss):
    '''
    The Object Balanced Loss encourages the object CoM to align with that of the gripper's CoM.
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
            ),
            robot_com = ti.Vector.field(3,F_DTYPE,shape=(),needs_grad=True),
            
            n_object_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            object_com = ti.Vector.field(3,F_DTYPE,shape=(),needs_grad=True),
        )
    
    def reset(self):
        super().reset()
        
    def compute_final_step_loss(self, s):
        self._compute_n_robot_particles()
        self._compute_n_object_particles()
        s_local = self.get_s_local(s)
        self._compute_com(s, s_local)
        self._compute_loss(s, s_local)

    def compute_final_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)
        self._compute_com.grad(s, s_local)
        
    @ti.func
    def _is_object(self,id):
        return id == self.obj_particle_id
    
    @ti.kernel
    def _compute_n_object_particles(self):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id):
                self.data['n_object_particles'][None] += ti.cast(1, F_DTYPE)
                
    @ti.kernel
    def _compute_com(self, s:I_DTYPE, s_local:I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            is_object = self._is_object(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                self.data['robot_com'][None] += self.env.sim.solver.x[s_local, p] / (self.data['n_robot_particles'][None])
            elif is_object:
                self.data['object_com'][None] += self.env.sim.solver.x[s_local, p] / (self.data['n_object_particles'][None])
                
    @ti.kernel
    def _compute_loss(self, s:I_DTYPE, s_local:I_DTYPE):
        self.data['loss'][s] = (self.data['robot_com'][None] - self.data['object_com'][None]).norm()
            