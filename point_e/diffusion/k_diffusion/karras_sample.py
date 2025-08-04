import torch.nn as nn
import torch as th

from typing import Union,Literal,Optional,Dict,Any

from .karras_denoiser import KarrasDenoiser
from .gaussian_to_karras_denoiser import GaussianToKarrasDenoiser
from ..gaussian_diffusion import GaussianDiffusion
from .karras_utils import get_sigmas_karras,to_d, get_ancestral_step

def karras_sample(*args, **kwargs):
    last = None
    for x in karras_sample_progressive(*args, **kwargs):
        last = x["x"]
    return last


def karras_sample_progressive(
    diffusion:Union[KarrasDenoiser,GaussianToKarrasDenoiser,GaussianDiffusion],
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    model_kwargs:Optional[Dict[str,Any]]=None,
    device:Optional[str]=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler:Literal['heun','dpm','ancestral']="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    guidance_scale=0.0,
):
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    x_T = th.randn(*shape, device=device) * sigma_max
    sample_fn = {"heun": sample_heun, "dpm": sample_dpm, "ancestral": sample_euler_ancestral}[
        sampler
    ]

    if sampler != "ancestral":
        sampler_args = dict(s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise)
    else:
        sampler_args = {}

    if isinstance(diffusion, KarrasDenoiser):
        def denoiser(x_t, sigma):
            _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs if model_kwargs is not None else {})
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
            return denoised
    elif isinstance(diffusion, GaussianDiffusion):
        model = GaussianToKarrasDenoiser(model, diffusion)

        def denoiser(x_t, sigma):
            _, denoised = model.denoise(
                x_t, sigma, clip_denoised=clip_denoised, model_kwargs=model_kwargs
            )
            return denoised
    else:
        raise NotImplementedError

    if guidance_scale != 0 and guidance_scale != 1:
        def guided_denoiser(x_t, sigma):
            x_t = th.cat([x_t, x_t], dim=0)
            sigma = th.cat([sigma, sigma], dim=0)
            x_0 = denoiser(x_t, sigma)
            cond_x_0, uncond_x_0 = th.split(x_0, len(x_0) // 2, dim=0)
            x_0 = uncond_x_0 + guidance_scale * (cond_x_0 - uncond_x_0)
            return x_0
    else:
        guided_denoiser = denoiser

    for obj in sample_fn(
        guided_denoiser,
        x_T,
        sigmas,
        progress=progress,
        **sampler_args,
    ):
        if isinstance(diffusion, GaussianDiffusion):
            yield diffusion.unscale_out_dict(obj)
        else:
            yield obj
            
@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, progress=False):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        yield {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "pred_xstart": denoised}
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + th.randn_like(x) * sigma_up
    yield {"x": x, "pred_xstart": x}


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        )
        eps = th.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        yield {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "pred_xstart": denoised}
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    yield {"x": x, "pred_xstart": denoised}


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        )
        eps = th.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        yield {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised}
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    yield {"x": x, "pred_xstart": denoised}