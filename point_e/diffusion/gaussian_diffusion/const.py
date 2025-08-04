from typing import Literal

DIFFUSION_MEAN_TYPE = Literal['x_prev','x_start','epsilon']
DIFFUSION_VAR_TYPE = Literal['learned','learned_range','fixed_large','fixed_small']
DIFFUSION_LOSS_TYPE = Literal['kl','rescaled_kl','mse','rescaled_mse']
