import taichi as ti
import torch

from .loss import Loss

from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

from typing import TYPE_CHECKING,List,Literal
if TYPE_CHECKING:
    from .lossset import LossSet
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class GrippingObjectLoss(Loss):
    '''
    The loss forces the robot to help make the object stay in its original position.
    '''
    
    def __init__(self, parent:'LossSet', env:'BaseEnv', obj_particle_id:int,obj_initial_pos:List[float]):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_object_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            robot_com = ti.Vector.field(3,F_DTYPE,shape=(),needs_grad=True),
            n_robot_particles = ti.field(
                dtype=F_DTYPE,
                shape=(),
                needs_grad=True
            ),
            initial_obj_x = ti.Vector.field(
                3,
                F_DTYPE,
                shape=(self.env.sim.solver.n_particles[None]),
                needs_grad=True
            )
        )
        self.obj_particle_id = obj_particle_id
        self.obj_initial_pos = ti.Vector(obj_initial_pos,F_DTYPE)

    def reset(self):
        super().reset()
        current_s = self.env.sim.solver.current_s
        current_s_local = self.env.sim.solver.get_cyclic_s(current_s)

        self._compute_misc(current_s_local)

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
    def _compute_misc(self,s:I_DTYPE):
        # TODO: Might need to chheck the validity of this loss function.
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id) and self.env.sim.solver.p_rho[p] > 0:
                self.data['n_object_particles'][None] += ti.cast(1, F_DTYPE)
                self.data['initial_obj_x'][p] = self.env.sim.solver.x[s, p] 
            elif self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                self.data['n_robot_particles'][None] += ti.cast(1, F_DTYPE)
                self.data['robot_com'][None] += self.env.sim.solver.x[s, p]
        if self.data['n_robot_particles'][None] > 0:
            self.data['robot_com'][None]/=self.data['n_robot_particles'][None]
                
    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id) and self.env.sim.solver.p_rho[p] > 0:
                x_mul = ti.Vector([1.,1.,1.], F_DTYPE) / self.data['n_object_particles'][None]
                diff = self.env.sim.solver.x[s_local, p] - (self.data['initial_obj_x'][p] - self.obj_initial_pos + self.data['robot_com'][None])
                p_loss = (diff*diff*x_mul).sum()
                self.data['loss'][s] += p_loss 