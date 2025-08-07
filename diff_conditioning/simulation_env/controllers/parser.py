import argparse

def augment_parser(parser:argparse.ArgumentParser):
    # Controller [General]
    parser.add_argument('--action-space', type=str, default='actuation',
                        choices=['actuation', 'particle_v', 'actuator_v'])
    parser.add_argument('--action-v-strength', type=float, default=1.)
    parser.add_argument('--controller-type', type=str, default='sin_wave_open_loop',
                        choices=['sin_wave_open_loop', 'pure_sin_wave_open_loop', 'random', 'trajopt',
                                 'mlp', 'sin_wave_closed_loop','all_on'])
    parser.add_argument('--controller-lr', type=float, default=0.003)
    
    # Controller [Sine wave]
    parser.add_argument('--n-sin-waves', type=int, default=4)
    parser.add_argument('--actuation-omega', type=float, nargs='+', default=[30.])
    
    # Controller [Trajectory optimization]
    parser.add_argument('--actuation-activation', type=str, default='linear',
                        choices=['tanh', 'softmax', 'linear'])
    
    # Controller [Pure Sine wave]
    parser.add_argument('--sin-omega-mul', type=float, default=10)
    
    # Controller [MLP]
    parser.add_argument('--controller-obs-names', type=str, nargs='+', default=['com', 'objective'])
    parser.add_argument('--controller-mlp-hidden-filters', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--controller-mlp-activation', type=str, default='Tanh')
    parser.add_argument('--controller-mlp-final-activation', type=str, default=None)
    
    # Controller [Closed-loop Sine wave]
    parser.add_argument('--closed-loop-n-sin-waves', type=int, default=4)
    parser.add_argument('--closed-loop-actuation-omega', type=float, nargs='+', default=[30.])
    parser.add_argument('--closed-loop-sinwave-obs-names', type=str, nargs='+', default=['com', 'objective'])
    parser.add_argument('--closed-loop-sinwave-hidden-filters', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--closed-loop-sinwave-activation', type=str, default='Tanh')

    return parser