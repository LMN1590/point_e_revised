""" Interface with designers. """
from .parser import augment_parser

from softzoo.envs.base_env import BaseEnv
from softzoo.configs.config_dataclass import DesignConfig

def make(args:DesignConfig, env:BaseEnv, torch_device):
    n_actuators = env.design_space.n_actuators
    if hasattr(args,'static_as_fixed'):
        static_as_fixed = args.static_as_fixed
    if hasattr(args, 'n_actuators'):
        n_actuators = args.n_actuators
    designer_kwargs = dict(
        env=env,
        n_actuators=n_actuators,
        lr=args.designer_lr,
        device=torch_device,
        static_as_fixed = static_as_fixed
    )
    if args.designer_type == 'mlp':
        from ..designer.mlp import MLP
        designer_cls = MLP
        designer_kwargs['coord_input_names'] = args.mlp_coord_input_names
        designer_kwargs['filters'] = args.mlp_filters
        designer_kwargs['activation'] = args.mlp_activation
        designer_kwargs['seed_meshes'] = args.mlp_seed_meshes
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
    elif args.designer_type == 'diff_cppn':
        from ..designer.diff_cppn import DiffCPPN
        designer_cls = DiffCPPN
        designer_kwargs['coord_input_names'] = args.cppn_coord_input_names
        designer_kwargs['seed_meshes'] = args.cppn_seed_meshes
        designer_kwargs['n_hiddens'] = args.cppn_n_hiddens
        designer_kwargs['activation_repeat'] = args.cppn_activation_repeat
        designer_kwargs['activation_options'] = args.cppn_activation_options
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
    elif args.designer_type == 'annotated_pcd':
        from ..designer.annotated_pcd import AnnotatedPCD
        designer_cls = AnnotatedPCD
        designer_kwargs['pcd_path'] = args.annotated_pcd_path
        designer_kwargs['n_voxels'] = args.annotated_pcd_n_voxels
        designer_kwargs['passive_geometry_mul'] = args.annotated_pcd_passive_geometry_mul
        designer_kwargs['passive_softness_mul'] = args.annotated_pcd_passive_softness_mul
    elif args.designer_type == 'sdf_basis':
        from ..designer.sdf_basis import SDFBasis
        designer_cls = SDFBasis
        designer_kwargs['pcd_paths'] = args.sdf_basis_pcd_paths
        designer_kwargs['mesh_paths'] = args.sdf_basis_mesh_paths
        designer_kwargs['passive_geometry_mul'] = args.sdf_basis_passive_geometry_mul
        designer_kwargs['passive_softness_mul'] = args.sdf_basis_passive_softness_mul
        designer_kwargs['init_coefs_geometry'] = args.sdf_basis_init_coefs_geometry
        designer_kwargs['init_coefs_softness'] = args.sdf_basis_init_coefs_softness
        designer_kwargs['init_coefs_actuator'] = args.sdf_basis_init_coefs_actuator
        designer_kwargs['init_coefs_actuator_direction'] = args.sdf_basis_init_coefs_actuator_direction
        designer_kwargs['use_global_coefs'] = args.sdf_basis_use_global_coefs
        designer_kwargs['n_voxels'] = args.sdf_basis_n_voxels
        designer_kwargs['coefs_activation'] = args.sdf_basis_coefs_activation
        designer_kwargs['actuator_mul'] = args.sdf_basis_actuator_mul
    elif args.designer_type == 'pbr':
        from ..designer.particle_based_repr import ParticleBasedRepresentation
        designer_cls = ParticleBasedRepresentation
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
    elif args.designer_type == 'vbr':
        from ..designer.voxel_based_repr import VoxelBasedRepresentation
        designer_cls = VoxelBasedRepresentation
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
        designer_kwargs['voxel_resolution'] = env.design_space.voxel_resolution
    elif args.designer_type == 'wass_barycenter':
        from ..designer.wass_barycenter import WassersteinBarycenter
        designer_cls = WassersteinBarycenter
        designer_kwargs['init_coefs_geometry'] = args.wass_barycenter_init_coefs_geometry
        designer_kwargs['init_coefs_actuator'] = args.wass_barycenter_init_coefs_actuator
        designer_kwargs['init_coefs_softness'] = args.wass_barycenter_init_coefs_softness
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
        designer_kwargs['voxel_resolution'] = env.design_space.voxel_resolution
        designer_kwargs['passive_geometry_mul'] = args.wass_barycenter_passive_geometry_mul
        designer_kwargs['passive_softness_mul'] = args.wass_barycenter_passive_softness_mul
    elif args.designer_type == 'loss_landscape_vbr':
        from ..designer.loss_landscape_vbr import LossLandscapeVBR
        designer_cls = LossLandscapeVBR
        designer_kwargs['voxel_resolution'] = env.design_space.voxel_resolution
        designer_kwargs['grid_index'] = args.loss_landscape_vbr_grid_index
        designer_kwargs['value_range'] = args.loss_landscape_vbr_value_range
        designer_kwargs['n_trials'] = args.loss_landscape_vbr_n_trials
        designer_kwargs['trial_type'] = args.loss_landscape_vbr_trial_type
    else:
        raise ValueError(f'Unrecognized designer type {args.designer_type}')
    designer = designer_cls(**designer_kwargs)

    return designer
