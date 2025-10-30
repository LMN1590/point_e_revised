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
        obj_particle_id:int, surface_threshold:float=0.02
    ):
        super().__init__(parent, env)
        self.obj_particle_id = obj_particle_id
        self.surface_threshold = surface_threshold
        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            surface_pts = ti.field(
                dtype=F_DTYPE,
                shape=(),
                needs_grad=True
            ),
            
            robot_min_dist = ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.n_particles[None]), needs_grad=True)
        )

    def reset(self):
        super().reset()
        self.data['robot_min_dist'].fill(1e4)
    
    def compute_final_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_min_dist(s, s_local)
        self._compute_loss(s, s_local)

    def compute_final_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)
        self._compute_min_dist.grad(s, s_local)
        
    @ti.func
    def _is_object(self,id):
        return id == self.obj_particle_id
    
    @ti.kernel
    def _compute_min_dist(self,s:I_DTYPE,s_local:I_DTYPE):
        for robot_id,obj_id in ti.ndrange(self.env.sim.solver.n_particles[None],self.env.sim.solver.n_particles[None]):
            id1 = self.env.sim.solver.particle_ids[robot_id]
            id2 = self.env.sim.solver.particle_ids[obj_id]
            is_robot = self.env.design_space.is_robot(id1) and (self.env.sim.solver.p_rho[robot_id] > 0)
            is_object = self._is_object(id2) and (self.env.sim.solver.p_rho[obj_id] > 0)
            if is_robot and is_object:
                pos1 = self.env.sim.solver.x[s_local,robot_id]
                pos2 = self.env.sim.solver.x[s_local,obj_id]
                dist = (pos1 - pos2).norm()
                self.data['robot_min_dist'][robot_id] = ti.atomic_min(self.data['robot_min_dist'][robot_id], dist)
    
    @ti.kernel
    def _compute_loss(self,s:I_DTYPE, s_local:I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                dist = self.data['robot_min_dist'][p]
                if dist<self.surface_threshold: self.data['surface_pts'][None] += 1.0
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                dist = self.data['robot_min_dist'][p]
                if dist<self.surface_threshold: self.data['loss'][s] += dist/(self.data['surface_pts'][None]+NORM_EPS)
        
        