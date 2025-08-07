from typing import Literal

from softzoo.configs.config_dataclass import FullConfig
from softzoo.envs.base_env import BaseEnv
from .loss_func.lossset import LossSet

def make_loss(args:FullConfig, env:BaseEnv, torch_device:Literal['cuda','cpu']):
    loss_configs = dict()
    loss_coefs = dict()
    for i, loss_type in enumerate(args.loss_types):
        if loss_type == 'FinalStepCoMLoss':
            loss_config = dict(x_mul=args.x_mul)
        elif loss_type == 'TrajectoryFollowingLoss':
            loss_config = dict(goal=args.goal)
        elif loss_type == 'PerStepCoVLoss':
            loss_config = dict(v_mul=args.v_mul)
        elif loss_type == 'AnimatedEMDLoss':
            loss_config = dict(mesh_dir=args.mesh_dir, substep_freq=args.substep_freq, 
                               mesh_num_points=args.mesh_num_points, final_target_idx=args.final_target_idx,
                               recenter_mesh_target=args.recenter_mesh_target)
        elif loss_type == 'VelocityFollowingLoss':
            loss_config = dict(v_mul=args.v_following_v_mul, mode=args.v_following_mode)
        elif loss_type == 'WaypointFollowingLoss':
            loss_config = dict()
        elif loss_type == 'RotationLoss':
            loss_config = dict(up_direction=args.rotation_up_direction)
        elif loss_type == 'ThrowingObjectLoss':
            loss_config = dict(x_mul=args.obj_x_mul, obj_particle_id=args.obj_particle_id)
        elif loss_type == 'ObjectVelocityLoss':
            loss_config = dict(v_mul=args.obj_v_mul, obj_particle_id=args.obj_particle_id)
        else:
            raise ValueError(f'Unrecognized loss type {loss_type}')
        loss_configs[loss_type] = loss_config
        loss_coefs[loss_type] = args.loss_coefs[i]
    loss_set = LossSet(env, loss_configs, loss_coefs)

    return loss_set