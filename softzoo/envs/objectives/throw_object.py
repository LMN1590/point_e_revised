import taichi as ti
import torch

from typing import TYPE_CHECKING

from .base import Base
from ...configs.config_dataclass import ThrowObjectConfig
from softzoo.utils.const import F_DTYPE,I_DTYPE,NORM_EPS

if TYPE_CHECKING:
    from ..base_env import BaseEnv

@ti.data_oriented
class ThrowObject(Base):
    def __init__(self, env:'BaseEnv', config:ThrowObjectConfig):
        super().__init__(env, config)
        self.config['reward_mode'] = self.config.get('reward_mode', 'per_step')
        assert self.config['reward_mode'] in ['per_step', 'final']
        
        self.config['forward_direction'] = self.config.get('forward_direction', [1., 0., 0.])
        
        self.max_episode_steps = self.config.get('max_episode_steps', self.env.max_steps)
        assert self.max_episode_steps <= self.env.max_steps, f'{self.max_episode_steps} is not <= {self.env.max_steps}'
        assert self.config['max_episode_steps'] != torch.inf, 'Maximal episode step is infinite'
        
        self.config['obj_particles_id'] = self.config.get('obj_particles_id',None)
        assert self.config['obj_particles_id'] is not None, "Specify the object's particles ID"
        
        self.data = dict(
            n_object_particles=ti.field(dtype=F_DTYPE, shape=()),
            object_initial_com = ti.Vector.field(
                3,
                F_DTYPE,
                shape=()
            ),
            cur_obj_avg = ti.Vector.field(
                3,
                F_DTYPE,
                shape=()
            )
        )
    
    def reset(self):
        for v in self.data.values():
            v.fill(0.)
        self._misc_compute()
        self.step_cnt = 0

    def get_obs(self, s):
        return None

    def get_reward(self, s):
        self.step_cnt += 1
        if self.config['reward_mode'] == 'per_step':
            s_local = self.env.sim.solver.get_cyclic_s(s)
            self._compute_x_avg(s_local,self.data['cur_obj_avg'])
            forward_dir = ti.Vector(self.config['forward_direction'])
            rew = ((self.data['cur_obj_avg'][None] - self.data['object_initial_com'][None]) * forward_dir).sum()
        elif self.config['reward_mode'] ==  'final':
            if self.step_cnt >= self.max_episode_steps:
                s_local = self.env.sim.solver.get_cyclic_s(s)
                self._compute_x_avg(s_local,self.data['cur_obj_avg'])
                forward_dir = ti.Vector(self.config['forward_direction'])
                rew = ((self.data['cur_obj_avg'][None] - self.data['object_initial_com'][None]) * forward_dir).sum()
            else:
                rew = 0.
        return float(rew)

    def get_done(self):
        return not (self.step_cnt < self.max_episode_steps)

    @property
    def obs_shape(self):
        return None
    
    # region Taichi Compute 
    def _misc_compute(self):
        self._compute_n_object_particles()
        self._compute_x_avg(0,self.data['object_initial_com'])
    
    @ti.func
    def _is_object(self, id):
        return id == self.config['obj_particles_id']
    
    @ti.kernel
    def _compute_n_object_particles(self):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id):
                self.data['n_object_particles'][None] += ti.cast(1, F_DTYPE)
    
    @ti.kernel
    def _compute_x_avg(self,s_local:I_DTYPE,ext_arr:ti.template()):
        # Require self.data['x_avg']
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_obj = self._is_object(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_obj:
                ext_arr[None] += self.env.sim.solver.x[s_local,p] / self.data['n_object_particles'][None]
    # endregion