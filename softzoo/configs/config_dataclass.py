from dataclasses import dataclass
from typing import Literal,List, Tuple, Any, Optional, Union, Dict
from typing_extensions import TypedDict

# region Simulator Config
@dataclass
class SolverConfig:
    use_dynamic_field: bool  # 
    needs_grad: bool  # 
    use_checkpointing: bool  # 
    dim: int  # world dimension
    padding: List[int]  # 
    quality: float  # determine grid size
    max_substeps: int  # 
    max_substeps_local: int  # 
    checkpoint_cache_device: Literal['numpy', 'torch_cpu', 'torch_gpu'] # 
    max_num_particles: int  # 
    default_dt: float  # time interval of a substep
    gravity: Tuple[float, float, float]  # 
    p_rho_0: float  # default particle density (mass / volume)
    E_0: float  # default Young's modulus
    nu_0: float  # default Poisson's ratio
    max_actuation: int  # maximal number of actuation
    base_active_materials: List[str]  # will be appended dynamically by items' materials

@dataclass
class ParticleGroupConfig:
    E_0: float
    nu_0: float
    mu_0: float
    lambd_0: float
    p_rho_0: float
    muscle_direction: Optional[List[float]]  # only used in SimpleMuscle
    active: bool
    max_coef_restitution: float
    
# region Primitive Config
@dataclass
class PrimitiveBaseConfig:
    type: str  # 'Primitive.Unidentified'
    spawn_order: int  # 0
    particle_id: int  # 0
    particle_info: ParticleGroupConfig  
    semantic_id: int  # 0
    item_id: int  # item id will be assigned dynamically
    
    material: str  # 'Elastic'
    sample_density: int  # 1
    density: float  # for rigid-body
    friction: float  # 0.9
    softness: float  # 0.

    initial_position: List[float]  # [0.3, 0.3]
    initial_rotation: List[float]  # [1., 0., 0., 0.]
    initial_velocity: List[float]  # for particles
    initial_twist: List[float]  # for rigid-body, 3-dim vector in 2D, 6-dim vector in 3D
    initial_wrench: List[float]  # [0., 0., 0.]
    
@dataclass
class BoxConfig(PrimitiveBaseConfig):
    type:str
    size:List[float]
    
@dataclass
class SphereConfig(PrimitiveBaseConfig):
    type:str
    radius:float

@dataclass
class MeshConfig(PrimitiveBaseConfig):
    type: str  # 'Primitive.Mesh'
    file_path: Optional[str]  # None
    scale: List[float]  # [1., 1., 1.]
    offset: List[float]  # Offset when loading the mesh; make sure triangles lie in [0, 0, 0] and [1, 1, 1]
    voxelizer_super_sample: int  # 2
    ground_height:float

@dataclass
class EllipsoidConfig(PrimitiveBaseConfig):
    type: str  # 'Primitive.Ellipsoid'
    radius: List[float]  # [0.1, 0.1, 0.1]
# endregion

# region Static Config
@dataclass
class DefaultStaticConfig:
    type:str
    semantic_id:int
    item_id:int # item id will be assigned dynamically

@dataclass
class BoundingBoxConfig(DefaultStaticConfig):
    pass

@dataclass
class FlatSurfaceConfig(DefaultStaticConfig):
    point:List[float]
    normal:List[float]
    surface:str
    friction:float
    pseudo_passive_velocity:List[float]
    dampen_coeff:float

@dataclass
class TerrainConfig(DefaultStaticConfig):
    surface:str
    friction:float
    pseudo_passive_velocity:List[float]
    min_height:float
    max_height:float
    resolution:int
    scale:float
    octaves: int
    persistence: float
    lacunarity: float
    repeat: int
    signed_dist_thresh: float
    dampen_coeff:float
# endregion

# region Objective Config
class ObjectiveForwardConfig(TypedDict):
    reward_mode: Literal['final_step_position', 'per_step_velocity']
    forward_direction: List[float]
    max_episode_steps: int

class ObjectiveTrajectoryConfig(TypedDict):
    start_velocity: List[float]
    end_position: List[float]
    end_position_rand_range: List[float]
    end_velocity: List[float]
    end_velocity_rand_range: List[float]
    end_substep: Optional[int]  # Should be set to env.max_steps during instantiation
    weight_norm: float
    weight_direction: float
    sigma_norm: float
    sigma_direction: float
    reward_mode: Literal['velocity_separate_exp','velocity_rmse','velocity_separate_linear','waypoint_sqsum','velocity_separate']

class ThrowObjectConfig(TypedDict):
    reward_mode: Literal['final_step_position', 'per_step_velocity']
    forward_direction: List[float]
    max_episode_steps: int
    obj_particles_id:int
# endregion

@dataclass
class DesignSpaceConfig:
    n_actuators:int # 10
    p_rho_lower_bound_mul: float #0.1
    initial_principle_direction: Optional[List[float]]
    voxel_resolution:List[int]
    
    base_shape: PrimitiveBaseConfig

# region Environment Config
@dataclass
class EnvCustomConfig:
    has_matter_on_ground:bool
    matter_id:int
    matter_materials: List[Literal['Water', 'Elastic', 'Snow', 'Sand', 'Stationary', 'SimpleMuscle', 'FakeRigid', 'DiffAquaMuscle', 'Mud', 'Plasticine']]
    matter_padding: List[int]
    matter_sample_density: int
    matter_semantic_id: int
    matter_thickness: float
    matter_youngs_modulus: float
    matter_depth:float #0.1
    matter_density: float #1.e+3
    randomize_terrain: bool
@dataclass
class EnvConfig:
    CUSTOM:EnvCustomConfig
    ITEMS:List[Union[PrimitiveBaseConfig,DefaultStaticConfig]]
    
    objective:Optional[Literal['trajectory_following','move_in_circles','move_forward','None']]
    objective_config:Union[ObjectiveForwardConfig,ObjectiveTrajectoryConfig,ThrowObjectConfig]
    
    use_renderer:bool
    frame_dt: float
    actuation_strength:float
    observation_space:List[str]
    
    design_space:str
    design_space_config:DesignSpaceConfig
    
    use_semantic_occupancy:bool
# endregion

# region Renderer Config
@dataclass
class GUIConfig:
    fps: int # 30
    title: str # "GUI Renderer"
    res: int # 512
    background_color: int # 0x112F41
    particle_colors: Tuple[int, int, int, int, int, int, int] # (0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00, 0x000000, 0xFF0000, 0x808080)
    static_component_color: int # 0x000000
    rigid_body_color: int # 0x000000
    circle_radius: float # 1.5

@dataclass
class GGUIConfig:
    title: str # "GGUI Renderer"
    tmp_fpath: str # "/tmp/tmp.png"
    offscreen_rendering: bool # True
    save_to_video: bool # True
    fps: int # 30
    res: Tuple[int, int] # (640, 480)
    ambient_light: Tuple[float, float, float] # (0.5, 0.5, 0.5)
    camera_position: List[float] # field(default_factory#lambda: [0.15, 0.75, -0.45])
    camera_lookat: List[float] # field(default_factory#lambda: [0.425, 0.309, 0.17])
    camera_fov: int # 55
    particle_radius: float # 0.01
    particle_colors: List[List[float]]
    particle_coloring_mode: Literal['material', 'particle_id', 'particle_density', 'actuation'] # "material"
    ground_surface_cmap: str # "Greys"
    meshify_particle_ids: List[int] 
    meshification_colors: List 
    background: Optional[str] # None
    background_color: Tuple[float, float, float] # (0.0, 0.0, 0.0)

@dataclass
class GLBodyInfo:
    draw_density: bool # False
    draw_diffuse: bool # False
    draw_ellipsoids: bool # True
    draw_points: bool # False
    needs_smoothing: bool # True
    particle_color: List[float]
    particle_radius: float # 0.001
    anisotropy_scale: float # 1.0

@dataclass
class GLConfig:
    save_to_video: bool # True
    dump_data: bool # False
    fps: int # 30
    res: Tuple[int, int] # (640, 480)
    camera_position: List[float] # [0.15, 0.75, -0.45]
    camera_lookat: List[float] # [0.425, 0.309, 0.17]
    camera_fov: int # 50
    draw_plane: bool # False
    light_position: List[float] # [0.5, 5.0, 0.5]
    light_lookat: List[float] # [0.5, 0.5, 0.49]
    light_fov: int # 50
    ground_surface_cmap: str # "texture/soil1_512x512.png"
    ground_surface_brightness_increase: int # 0
    background: Optional[str] # "skybox/vary_sky.jpg"
    background_brightness_increase: int # 0
    tile_texture: bool # False
    msaa_samples: int # 8
    anisotropy_scale: float # 1.0
    smoothing: float # 0.5
    rendering_scale: float # 2.0
    fluid_rest_distance: float # 0.0125
    gl_color_gamma: float # 3.5
    bodies_info: Dict[str, GLBodyInfo] # field(default_factory#dict)

@dataclass
class RendererConfig:
    type:Literal['ggui','gui','gl']
    
    GUI:Optional[GUIConfig]
    GGUI:Optional[GGUIConfig]
    GL:Optional[GLConfig]
# endregion

@dataclass
class SimulatorConfig:
    SIMULATOR:SolverConfig
    ENVIRONMENT:EnvConfig
    RENDERER:RendererConfig
# endregion
    


# region Input Config
@dataclass
class BaseControllerConfig:
    # Controller [General]
    action_space: Literal['actuation', 'particle_v', 'actuator_v'] # 'actuation'
    action_v_strength: float # 1.
    controller_type: str # 'sin_wave_open_loop'
    controller_lr: float # 0.003

    # Controller [Sine wave]
    n_sin_waves: int # 4
    actuation_omega: List[float] # field(default_factory#lambda: [30.])

    # Controller [Trajectory optimization]
    actuation_activation: str # 'linear'

    # Controller [Pure Sine wave]
    sin_omega_mul: float # 10.

    # Controller [MLP]
    controller_obs_names: List[str] # field(default_factory#lambda: ['com', 'objective'])
    controller_mlp_hidden_filters: List[int] # field(default_factory#lambda: [32, 32])
    controller_mlp_activation: str # 'Tanh'
    controller_mlp_final_activation: Optional[str] # None

    # Controller [Closed-loop Sine wave]
    closed_loop_n_sin_waves: int # 4
    closed_loop_actuation_omega: List[float] # field(default_factory#lambda: [30.])
    closed_loop_sinwave_obs_names: List[str] # field(default_factory#lambda: ['com', 'objective'])
    closed_loop_sinwave_hidden_filters: List[int] # field(default_factory#lambda: [32, 32])
    closed_loop_sinwave_activation: str # 'Tanh'
    
    # Controller [AllOn]
    active:bool

@dataclass
class DesignConfig:
    # Designer [General]
    designer_type: str # 'mlp'
    designer_lr: float # 0.003
    designer_geometry_offset: float # 0.5
    designer_softness_offset: float # 0.5

    # Designer [MLP]
    mlp_coord_input_names: List[str] # field(default_factory#lambda: ['x', 'y', 'z', 'd_xy', 'd_yz', 'd_xz', 'd_xyz'])
    mlp_filters: List[int] # field(default_factory#lambda: [32, 32])
    mlp_activation: str # 'Tanh'
    mlp_seed_meshes: List[str] # field(default_factory#list)

    # Designer [Diff-CPPN]
    cppn_coord_input_names: List[str] # field(default_factory#lambda: ['x', 'y', 'z', 'd_xy', 'd_yz', 'd_xz', 'd_xyz'])
    cppn_seed_meshes: List[str] # field(default_factory#list)
    cppn_n_hiddens: int # 3
    cppn_activation_repeat: int # 10
    cppn_activation_options: List[str] # field(default_factory#lambda: ['sin', 'sigmoid'])

    # Designer [Annotated-PCD]
    annotated_pcd_path: Optional[str] # None
    annotated_pcd_n_voxels: int # 60
    annotated_pcd_passive_softness_mul: float # 10.0
    annotated_pcd_passive_geometry_mul: float # 0.5

    # Designer [SDF Basis]
    sdf_basis_pcd_paths: List[str] # field(default_factory#list)
    sdf_basis_mesh_paths: List[str] # field(default_factory#list)
    sdf_basis_passive_softness_mul: float # 10.0
    sdf_basis_passive_geometry_mul: float # 0.5
    sdf_basis_init_coefs_geometry: Optional[List[float]] # None
    sdf_basis_init_coefs_softness: Optional[List[float]] # None
    sdf_basis_init_coefs_actuator: Optional[List[float]] # None
    sdf_basis_init_coefs_actuator_direction: Optional[List[float]] # None
    sdf_basis_use_global_coefs: bool # False
    sdf_basis_n_voxels: int # 60
    sdf_basis_coefs_activation: str # 'linear'
    sdf_basis_actuator_mul: float # 1.0

    # Designer [Wasserstein Barycenter]
    wass_barycenter_init_coefs_geometry: Optional[List[float]] # None
    wass_barycenter_init_coefs_actuator: Optional[List[float]] # None
    wass_barycenter_init_coefs_softness: Optional[List[float]] # None
    wass_barycenter_passive_softness_mul: float # 10.0
    wass_barycenter_passive_geometry_mul: float # 0.5

    # Designer [Loss Landscape Voxel-based Representation]
    loss_landscape_vbr_grid_index: List[int] # field(default_factory#lambda: [0, 0, 0])
    loss_landscape_vbr_value_range: List[float] # field(default_factory#lambda: [0.0, 1.0])
    loss_landscape_vbr_n_trials: int # 10
    loss_landscape_vbr_trial_type: str # 'geometry'
    
    static_as_fixed:bool
    
    gen_pointe_bounding_box: Optional[Dict[Literal['max','mean','min'],List[float]]]

@dataclass
class FullConfig(BaseControllerConfig,DesignConfig):
    # Seeds and Logging
    seed: int # 100
    torch_seed: int # 100
    render_every_iter: int # 10
    save_every_iter: int # 10
    log_every_iter: int # 1
    log_reward: bool # False

    # Load / Save Modules
    load_args: Optional[str] # None
    load_controller: Optional[str] # None
    load_rl_controller: Optional[str] # None
    load_designer: Optional[str] # None
    save_designer: bool # False
    save_controller: bool # False
    eval: bool # False

    # Environment
    out_dir: str # '/tmp/tmp'
    non_taichi_device: str # 'torch_cpu'
    env: str # 'land_environment'
    env_config_file: str # 'fixed_plain.yaml'
    objective_reward_mode: Optional[str] # None
    dump_rendering_data: bool # False
    custom_gravity: bool

    # Optimization
    n_iters: int # 1
    n_frames: int # 10
    loss_types: List[str] # field(default_factory#lambda: ['FinalStepCoMLoss'])
    loss_coefs: List[float] # field(default_factory#lambda: [1.])
    optimize_designer: bool # False
    set_design_types: List[Literal['geometry', 'softness', 'actuator', 'actuator_direction','is_passive_fixed']] # field(default_factory#lambda)
    optimize_design_types: List[Literal['geometry', 'softness', 'actuator']] # field(default_factory#lambda: )
    optimize_controller: bool # False

    # Final Step CoM Loss
    x_mul: List[float] # field(default_factory#lambda: [1., 0., 0.])

    # Trajectory Following Loss
    goal: List[float] # field(default_factory#lambda: [0.8, 0., 0., 1700])

    # Per Step CoV Loss
    v_mul: List[float] # field(default_factory#lambda: [1., 0., 0.])

    # Animated EMD Loss
    mesh_dir: str # './local/meshes/fantasy_horse'
    substep_freq: int # 100
    mesh_num_points: int # 5000
    final_target_idx: Optional[int] # None
    recenter_mesh_target: bool # False

    # Velocity Following Loss
    v_following_v_mul: List[float] # field(default_factory#lambda: [1., 1., 1.])
    v_following_mode: int # 0

    # Rotation Loss
    rotation_up_direction: List[float] # field(default_factory#lambda: [0., 1., 0.])

    # Throwing Object Loss
    obj_x_mul: List[float] # field(default_factory#lambda: [1., 0., 0.])
    obj_particle_id: int # 2

    # Object Velocity Loss
    obj_v_mul: List[float] # field(default_factory#lambda: [1., 0., 0.])
    
    # Gripping Object Loss
    obj_initial_pos:List[float]
    
    # Contact Loss
    surface_threshold: float # 0.02
    # Utils
    device_memory_fraction:float
    design_device: str
    fixed_v:str
    num_fingers:int
# endregion