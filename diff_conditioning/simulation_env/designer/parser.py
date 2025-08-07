import argparse

def augment_parser(parser:argparse.ArgumentParser):
    # Designer [General]
    parser.add_argument('--designer-type', type=str, default='mlp')
    parser.add_argument('--designer-lr', type=float, default=0.003)
    parser.add_argument('--designer-geometry-offset', type=float, default=0.5)
    parser.add_argument('--designer-softness-offset', type=float, default=0.5)
    # Designer [MLP]
    parser.add_argument('--mlp-coord-input-names', type=str, nargs='+', default=['x', 'y', 'z', 'd_xy', 'd_yz', 'd_xz', 'd_xyz'])
    parser.add_argument('--mlp-filters', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--mlp-activation', type=str, default='Tanh')
    parser.add_argument('--mlp-seed-meshes', type=str, nargs='+', default=[])
    # Designer [Diff-CPPN]
    parser.add_argument('--cppn-coord-input-names', type=str, nargs='+', default=['x', 'y', 'z', 'd_xy', 'd_yz', 'd_xz', 'd_xyz'])
    parser.add_argument('--cppn-seed-meshes', type=str, nargs='+', default=[])
    parser.add_argument('--cppn-n-hiddens', type=int, default=3)
    parser.add_argument('--cppn-activation-repeat', type=int, default=10)
    parser.add_argument('--cppn-activation-options', type=str, nargs='+', default=['sin', 'sigmoid'])
    # Designer [Annotated-PCD]
    parser.add_argument('--annotated-pcd-path', type=str, default=None)
    parser.add_argument('--annotated-pcd-n-voxels', type=int, default=60)
    parser.add_argument('--annotated-pcd-passive-softness-mul', type=float, default=10)
    parser.add_argument('--annotated-pcd-passive-geometry-mul', type=float, default=0.5)
    # Designer [SDF Basis]
    parser.add_argument('--sdf-basis-pcd-paths', type=str, nargs='+', default=[])
    parser.add_argument('--sdf-basis-mesh-paths', type=str, nargs='+', default=[])
    parser.add_argument('--sdf-basis-passive-softness-mul', type=float, default=10)
    parser.add_argument('--sdf-basis-passive-geometry-mul', type=float, default=0.5)
    parser.add_argument('--sdf-basis-init-coefs-geometry', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-init-coefs-softness', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-init-coefs-actuator', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-init-coefs-actuator-direction', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-use-global-coefs', action='store_true', default=False)
    parser.add_argument('--sdf-basis-n-voxels', type=int, default=60)
    parser.add_argument('--sdf-basis-coefs-activation', type=str, default='linear')
    parser.add_argument('--sdf-basis-actuator-mul', type=float, default=1.)
    # Designer [Particle-based Representation]
    pass
    # Designer [Voxel-based Representation]
    pass
    # Designer [Wasserstein Barycenter]
    parser.add_argument('--wass-barycenter-init-coefs-geometry', type=float, nargs='+', default=None)
    parser.add_argument('--wass-barycenter-init-coefs-actuator', type=float, nargs='+', default=None)
    parser.add_argument('--wass-barycenter-init-coefs-softness', type=float, nargs='+', default=None)
    parser.add_argument('--wass-barycenter-passive-softness-mul', type=float, default=10)
    parser.add_argument('--wass-barycenter-passive-geometry-mul', type=float, default=0.5)
    # Designer [Loss Lanscape Voxel-based Representation] NOTE: used for study diff-physics
    parser.add_argument('--loss-landscape-vbr-grid-index', type=int, nargs='+', default=[0, 0, 0])
    parser.add_argument('--loss-landscape-vbr-value-range', type=float, nargs='+', default=[0., 1.])
    parser.add_argument('--loss-landscape-vbr-n-trials', type=int, default=10)
    parser.add_argument('--loss-landscape-vbr-trial-type', type=str, default='geometry')
    
    parser.add_argument('--static-as-fixed',action='store_true', default=False)

    return parser