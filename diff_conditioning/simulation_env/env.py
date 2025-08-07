from typing import Literal,Type

from softzoo.configs.config_dataclass import FullConfig

def make_env(args:FullConfig):
    if args.env == 'land_environment':
        from softzoo.envs.land_environment import LandEnvironment
        env_cls = LandEnvironment
    elif args.env == 'aquatic_environment':
        from softzoo.envs.aquatic_environment import AquaticEnvironment
        env_cls = AquaticEnvironment
    elif args.env == 'dummy_env':
        from softzoo.envs.dummy_env import DummyEnv
        env_cls = DummyEnv
    elif args.env == 'manipulation_environment':
        from softzoo.envs.manipulation_environment import ManipulationEnvironment
        env_cls = ManipulationEnvironment
    else:
        raise NotImplementedError

    cfg_kwargs = dict()
    if args.render_every_iter > 0:
        cfg_kwargs['ENVIRONMENT.use_renderer'] = True
    if args.objective_reward_mode not in [None, 'None']:
        cfg_kwargs['ENVIRONMENT.objective_config.reward_mode'] = args.objective_reward_mode
    if args.dump_rendering_data:
        cfg_kwargs['RENDERER.GL.dump_data'] = True
    env_kwargs = dict(
        cfg_file=args.env_config_file,
        out_dir=args.out_dir,
        device=args.non_taichi_device,
        cfg_kwargs=cfg_kwargs,
    )
    env = env_cls(**env_kwargs)
    env.initialize()

    if args.render_every_iter > 0: assert env.has_renderer

    return env