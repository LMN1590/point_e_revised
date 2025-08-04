import numpy as np
import torch as th
import torch.nn as nn

from .karras_utils import append_dims
from ..gaussian_diffusion import GaussianDiffusion

class GaussianToKarrasDenoiser:
    def __init__(self, model:nn.Module, diffusion:GaussianDiffusion):
        from scipy import interpolate

        self.model = model
        self.diffusion = diffusion
        self.alpha_cumprod_to_t = interpolate.interp1d(
            diffusion.alphas_cumprod, np.arange(0, diffusion.num_timesteps)
        )

    def sigma_to_t(self, sigma):
        alpha_cumprod = 1.0 / (sigma**2 + 1)
        if alpha_cumprod > self.diffusion.alphas_cumprod[0]:
            return 0
        elif alpha_cumprod <= self.diffusion.alphas_cumprod[-1]:
            return self.diffusion.num_timesteps - 1
        else:
            return float(self.alpha_cumprod_to_t(alpha_cumprod))

    def denoise(self, x_t, sigmas, clip_denoised=True, model_kwargs=None):
        t = th.tensor(
            [self.sigma_to_t(sigma) for sigma in sigmas.cpu().numpy()],
            dtype=th.long,
            device=sigmas.device,
        )
        c_in = append_dims(1.0 / (sigmas**2 + 1) ** 0.5, x_t.ndim)
        out = self.diffusion.p_mean_variance(
            self.model, x_t * c_in, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        return None, out["pred_xstart"]