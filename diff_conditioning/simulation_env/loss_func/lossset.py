import taichi as ti

from typing import Dict,TYPE_CHECKING,Type,List,Callable

from softzoo.utils.const import I_DTYPE,F_DTYPE

from .loss import Loss
from .animated_emd_loss import AnimatedEMDLoss
from .final_step_com_loss import FinalStepCoMLoss
from .object_velocity_loss import ObjectVelocityLoss
from .perstep_cov_loss import PerStepCoVLoss
from .rotation_loss import RotationLoss
from .throwing_obj_loss import ThrowingObjectLoss
from .trajectory_following_loss import TrajectoryFollowingLoss
from .velocity_following_loss import VelocityFollowingLoss
from .waypoint_following_loss import WaypointFollowingLoss
from .gripping_obj_loss import GrippingObjectLoss

if TYPE_CHECKING:
    from softzoo.envs.base_env import BaseEnv

@ti.data_oriented
class LossSet:
    def __init__(
        self, env:'BaseEnv',
        loss_configs: Dict[str,Dict],
        loss_coefs: Dict[str,float]
    ):
        self.env = env
        
        self.total_loss = ti.field(
            dtype = self.env.sim.solver.f_dtype,
            shape = (self.env.sim.solver.max_substeps),
            needs_grad= True
        )
        self.loss_names = loss_configs.keys()
        self.losses:List[Loss] = [] # use list to be accessed in taichi kernel
        self.loss_coefs = []
        for k,v in loss_configs.items():
            loss_cls:Type[Loss] = globals()[k]
            self.losses.append(loss_cls(
                parent = self,
                env = env,
                **v
            ))
            self.loss_coefs.append(loss_coefs[k])
        self.shared_data = dict()  # TODO: use shared data to avoid recomputation
        
    def compute_loss(
        self, 
        post_substep_grad_fn:List[Callable] = [],
        compute_grad= False,
        grad_names:Dict[int,List[str]] = dict()
    ):
        # Instantiate a cache for desired gradients
        grad = {k: {vv: None for vv in v} for k, v in grad_names.items()}
        grad_s = grad_names.keys()
        
        # Forward
        latest_s = self.env.sim.solver.get_latest_s(consider_cycle=False)
        norm_factor = 1. # / latest_s # TODO: do we need normalization?!
        
        # Backward
        if compute_grad:
            for s in range(latest_s,0,-1):
                def pre_grad_fn():
                    # Compute Loss
                    for loss in self.losses:
                        if s == latest_s:
                            loss.compute_final_step_loss(s)
                        else:
                            loss.compute_per_step_loss(s)
                    self.accumulate_loss(s, norm_factor)
                    
                    # Compute loss gradient
                    self.total_loss.grad[s] = 1.
                    self.accumulate_loss.grad(s, norm_factor)
                    for loss in self.losses:
                        if s==latest_s:
                            loss.compute_final_step_grad(s)
                        else:
                            loss.compute_per_step_grad(s)
                sm1 = s-1 # NOTE: to obtain gradient at s, we need to call substep_grad(s-1)
                action_sm1 = self.env.sim.solver.checkpoint_cache['act_buffer'][sm1]
                dt_sm1 = self.env.sim.solver.checkpoint_cache['dt'][sm1]
                self.env.sim.solver.substep_grad(action_sm1,dt=dt_sm1,pre_grad_fn=pre_grad_fn)
                
                sm1_local = self.env.sim.solver.current_s_local # NOTE: substep_grad decrement s
                
                for grad_fn in post_substep_grad_fn:
                    grad_fn(sm1,sm1_local)
                    
                if sm1 in grad_s:
                    assert sm1_local == self.env.sim.solver.get_cyclic_s(sm1)
                    for grad_k in grad[sm1].keys():
                        if grad_k[:20] == 'self.env.sim.solver.':
                            var_name = grad_k[20:]
                            grad[sm1][grad_k] = self.env.sim.device.clone(self.env.sim.apply('get', var_name + '.grad', s=sm1_local))
                        elif grad_k[:22] == 'self.env.design_space.':
                            var_name = grad_k[22:]
                            assert var_name == 'v_buffer'
                            grad[sm1][grad_k] = self.env.design_space.get_v_buffer_grad(s=sm1)
                        else:
                            raise ValueError(f'Unrecognized gradient name {grad_k}')
            
            # Compute gradient in design space
            self.env.design_space.set_design_grad(None)
            for s_none in grad.keys():
                if s_none is not None: # time-invariant data
                    continue
                for grad_k in grad[s_none].keys():
                    if grad_k[:20] == 'self.env.sim.solver.':
                        var_name = grad_k[20:]
                        grad[s_none][grad_k] = self.env.sim.device.clone(self.env.sim.apply('get', var_name + '.grad'))
                    elif grad_k[:29] == 'self.env.design_space.buffer.':
                        grad[s_none][grad_k] = self.env.sim.device.to_ext(eval(grad_k + '.grad'))
                    else:
                        raise ValueError(f'Unrecognized gradient name {grad_k}')
        else:
            for s in range(latest_s, 0, -1):
                def pre_grad_fn():
                    # Compute loss
                    for loss in self.losses:
                        if s == latest_s:
                            loss.compute_final_step_loss(s)
                        else:    
                            loss.compute_per_step_loss(s)
                    self.accumulate_loss(s, norm_factor)

                sm1 = s - 1 # NOTE: to obtain gradient at s, we need to call substep_grad(s-1)
                action_sm1 = self.env.sim.solver.checkpoint_cache['act_buffer'][sm1]
                dt_sm1 = self.env.sim.solver.checkpoint_cache['dt'][sm1]
                self.env.sim.solver.substep_grad(action_sm1, dt=dt_sm1, pre_grad_fn=pre_grad_fn, compute_grad=False)

        # print(latest_s, self.total_loss.to_numpy()[:latest_s+1].sum()) # DEBUG

        return self.total_loss.to_numpy()[:latest_s+1], grad
                    
    def reset(self,loss_reset_kwargs):
        for k, v in zip(self.loss_names, self.losses):
            v.reset(**loss_reset_kwargs[k])

        self.total_loss.fill(0.)
        self.total_loss.grad.fill(0.)
    
    @ti.kernel
    def accumulate_loss(self,s:I_DTYPE, norm_factor:F_DTYPE):
        for i in ti.static(range(len(self.losses))):
            self.total_loss[s] += self.loss_coefs[i] * self.losses[i].data['loss'][s] * norm_factor