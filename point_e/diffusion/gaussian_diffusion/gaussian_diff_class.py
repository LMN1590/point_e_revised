import numpy as np
import torch as th
import torch.nn as nn

from typing import Sequence, Optional, Literal, Callable, Dict, Tuple, Union, Any
from tqdm import tqdm
import logging
from logger import TENSORBOARD_LOGGER

from .const import DIFFUSION_LOSS_TYPE,DIFFUSION_MEAN_TYPE,DIFFUSION_VAR_TYPE
from ..diff_utils import _extract_into_tensor,normal_kl,discretized_gaussian_log_likelihood,mean_flat,approx_standard_normal_cdf

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D array of betas for each diffusion timestep from T to 1.
    :param model_mean_type: a string determining what the model outputs.
    :param model_var_type: a string determining how variance is output.
    :param loss_type: a string determining the loss function to use.
    :param discretized_t0: if True, use discrete gaussian loss for t=0. Only
                           makes sense for images.
    :param channel_scales: a multiplier to apply to x_start in training_losses
                           and sampling functions.
    """

    def __init__(
        self,
        *,
        betas: Sequence[float],
        model_mean_type: DIFFUSION_MEAN_TYPE,
        model_var_type: DIFFUSION_VAR_TYPE,
        loss_type: DIFFUSION_LOSS_TYPE,
        k:int,
        condition_threshold:int,
        discretized_t0: bool = False,
        channel_scales: Optional[np.ndarray] = None,
        channel_biases: Optional[np.ndarray] = None,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.discretized_t0 = discretized_t0
        self.channel_scales = channel_scales
        self.channel_biases = channel_biases
        self.k = k
        self.condition_threshold = condition_threshold

        # region Alpha and Beta Calculation
        # Use float64 for accuracy.
        betas_np = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas_np.shape) == 1, "betas must be 1-D"
        assert (betas_np > 0).all() and (betas_np <= 1).all()
        
        self.num_timesteps = int(betas_np.shape[0])
        
        alphas = 1.0 - betas_np
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        # endregion
        
        # region Posterior Calculation
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        # endregion
        
    def get_sigmas(self, t:th.Tensor):
        return _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, t.shape) 
    
    # region Q Function Operations
    def q_mean_variance(self, x_start:th.Tensor, t:th.Tensor):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start:th.Tensor, t:th.Tensor, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    def q_posterior_mean_variance(self, x_start:th.Tensor, x_t:th.Tensor, t:th.Tensor):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    # endregion
    
    # region Prediction Operations
    def p_mean_variance(
        self, model:nn.Module, x:th.Tensor, t:th.Tensor, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        # Calculate the Log Variance
        if self.model_var_type in ["learned", "learned_range"]:
            assert model_output.shape == (B, C * 2, *x.shape[2:]) # C*2: C_1 is the content of each channel, C_2 is the variance for each channel.
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == "learned":
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                "fixed_large": (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                "fixed_small": (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        model_eps = None
        if self.model_mean_type == "x_prev":
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
            model_eps = self._predict_eps_from_xstart(x, t, pred_xstart)
        elif self.model_mean_type in ["x_start", "epsilon"]:
            if self.model_mean_type == "x_start":
                pred_xstart = process_xstart(model_output)
                model_eps = self._predict_eps_from_xstart(x, t, pred_xstart)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
                model_eps = model_output
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
            "eps": model_eps
        }
    
    def _predict_xstart_from_eps(self, x_t:th.Tensor, t:th.Tensor, eps:th.Tensor):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t:th.Tensor, t:th.Tensor, xprev:th.Tensor):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t:th.Tensor, t:th.Tensor, pred_xstart:th.Tensor):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    # endregion
    
    # region Condtion Operation
    def condition_mean(self, cond_fn:Callable[...,th.Tensor], p_mean_var:Dict[str,th.Tensor], x:th.Tensor, t:th.Tensor, local_iter:int, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        One of the desciprtion for this function can be found in Algo 1 of https://arxiv.org/pdf/2105.05233
        """
        condition_kwargs = model_kwargs.copy()
        condition_kwargs['diffusion'] = self if "diffusion" not in condition_kwargs else condition_kwargs['diffusion']
        condition_kwargs['p_mean_var'] = p_mean_var
        condition_kwargs['local_iter'] = local_iter
        
        gradient = cond_fn(x, t, **condition_kwargs)
        mean_modifier = p_mean_var["variance"] * gradient.float()
        new_mean = p_mean_var["mean"].float() + mean_modifier
        TENSORBOARD_LOGGER.log_scalar("Conditioning_DDPM/Original_Mean",p_mean_var["mean"].float().reshape(-1).norm(2))
        TENSORBOARD_LOGGER.log_scalar("Conditioning_DDPM/Mean_Modifier",mean_modifier.reshape(-1).norm(2))
        return new_mean

    def condition_score(self, cond_fn:Callable[...,th.Tensor], p_mean_var:Dict[str,th.Tensor], x:th.Tensor, t:th.Tensor, local_iter:int, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        
        Apply the gradient to eps instead of the mean
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = p_mean_var["eps"]
        condition_kwargs = model_kwargs.copy()
        condition_kwargs['diffusion'] = self if "diffusion" not in condition_kwargs else condition_kwargs['diffusion']
        condition_kwargs['p_mean_var'] = p_mean_var
        condition_kwargs['local_iter'] = local_iter
        eps_modifier = (1 - alpha_bar).sqrt() * cond_fn(x, t, **condition_kwargs)
        eps = eps - eps_modifier
        
        TENSORBOARD_LOGGER.log_scalar("Conditioning_DDIM/Original_Eps",eps.reshape(-1).norm(2))
        TENSORBOARD_LOGGER.log_scalar("Conditioning_DDIM/Eps_Modifier",eps_modifier.reshape(-1).norm(2))

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        out['eps'] = eps
        return out
    # endregion
    
    # region Sampling Operations
    def p_sample(
        self,
        model:nn.Module,
        x:th.Tensor,
        t:th.Tensor,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        noise = th.randn_like(x)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        
        if cond_fn is not None and t[0]<=self.condition_threshold:
            TENSORBOARD_LOGGER.log_scalar("Overall_DDPM_Original/Predicted_Mean_xt_L2Norm",out["mean"].reshape(-1).norm(2))
            TENSORBOARD_LOGGER.log_scalar("Overall_DDPM_Original/Sample_xt-1_L2Norm",sample.reshape(-1).norm(2))
            sample = x
            for local_iter in tqdm(range(self.k),desc = f"Current sampling step {t[0].item()}",position=1,leave=False):
                logging.info(f"Current sampling step {t[0].item()} - local iter {local_iter}")
                out["mean"] = self.condition_mean(
                    cond_fn, out, sample, t, 
                    local_iter = local_iter, model_kwargs=model_kwargs
                )
                noise = th.randn_like(x)
                sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                TENSORBOARD_LOGGER.log_scalar("Overall_DDPM/Predicted_Mean_xt_L2Norm",out["mean"].reshape(-1).norm(2))
                TENSORBOARD_LOGGER.log_scalar("Overall_DDPM/Sample_xt-1_L2Norm",sample.reshape(-1).norm(2))
                TENSORBOARD_LOGGER.increment()
            
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def p_sample_loop(
        self,
        model:nn.Module,
        shape:Tuple,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        temp=1.0,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            temp=temp,
        ):
            final = sample
        return final["sample"]
    
    def p_sample_loop_progressive(
        self,
        model:nn.Module,
        shape:Tuple,
        noise:Optional[th.Tensor]=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        temp=1.0,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise.to(device)
        else:
            img = th.randn(*shape, device=device) * temp
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield self.unscale_out_dict(out)
                img = out["sample"]
    # endregion
    
    # region DDIM Sampling Operations
    def ddim_sample(
        self,
        model:nn.Module,
        x:th.Tensor,
        t:th.Tensor,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred_eps_modifier = th.sqrt(1 - alpha_bar_prev - sigma**2) * out['eps']
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + mean_pred_eps_modifier
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        
        if cond_fn is not None and t[0]<=self.condition_threshold:
            TENSORBOARD_LOGGER.log_scalar("Overall_DDIM_Original/Eps_Modifier_xt_L2Norm",mean_pred_eps_modifier.reshape(-1).norm(2))
            TENSORBOARD_LOGGER.log_scalar("Overall_DDIM_Original/Modified_Predicted_Mean_xt_L2Norm",mean_pred.reshape(-1).norm(2))
            TENSORBOARD_LOGGER.log_scalar("Overall_DDIM_Original/Sample_xt-1_L2Norm",sample.reshape(-1).norm(2))
            TENSORBOARD_LOGGER.log_scalar("Overall_DDIM_Original/Eps_L2Norm",out['eps'].reshape(-1).norm(2))
            sample = x
            
            for local_iter in tqdm(range(self.k),desc = f"Current iter {t[0].item()}",position=1,leave=False):
                out = self.condition_score(
                    cond_fn, out, sample, t, 
                    local_iter = local_iter, model_kwargs=model_kwargs
                )
                alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
                alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
                sigma = (
                    eta
                    * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * th.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                # Equation 12.
                noise = th.randn_like(x)
                mean_pred_eps_modifier = th.sqrt(1 - alpha_bar_prev - sigma**2) * out['eps']
                mean_pred = (
                    out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                    + mean_pred_eps_modifier
                )
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )  # no noise when t == 0
                sample = mean_pred + nonzero_mask * sigma * noise
                TENSORBOARD_LOGGER.log_scalar("Overall_DDIM/Eps_L2Norm",out['eps'].reshape(-1).norm(2))
                TENSORBOARD_LOGGER.log_scalar("Overall_DDIM/Eps_Modifier_xt_L2Norm",mean_pred_eps_modifier.reshape(-1).norm(2))
                TENSORBOARD_LOGGER.log_scalar("Overall_DDIM/Modified_Predicted_Mean_xt_L2Norm",mean_pred.reshape(-1).norm(2))
                TENSORBOARD_LOGGER.log_scalar("Overall_DDIM/Sample_xt-1_L2Norm",sample.reshape(-1).norm(2))
                TENSORBOARD_LOGGER.increment()
            

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    # def ddim_reverse_sample(
    #     self,
    #     model:nn.Module,
    #     x:th.Tensor,
    #     t:th.Tensor,
    #     clip_denoised=False,
    #     denoised_fn=None,
    #     cond_fn=None,
    #     model_kwargs=None,
    #     eta=0.0,
    # ):
    #     """
    #     Sample x_{t+1} from the model using DDIM reverse ODE.
    #     """
    #     assert eta == 0.0, "Reverse ODE only for deterministic path"
    #     out = self.p_mean_variance(
    #         model,
    #         x,
    #         t,
    #         clip_denoised=clip_denoised,
    #         denoised_fn=denoised_fn,
    #         model_kwargs=model_kwargs,
    #     )
    #     if cond_fn is not None:
    #         out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
    #     # Usually our model outputs epsilon, but we re-derive it
    #     # in case we used x_start or x_prev prediction.
    #     eps = (
    #         _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
    #         - out["pred_xstart"]
    #     ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
    #     alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

    #     # Equation 12. reversed
    #     mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps

    #     return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
    
    def ddim_sample_loop(
        self,
        model:nn.Module,
        shape:Tuple,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        temp=1.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            temp=temp,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model:nn.Module,
        shape:Tuple,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        temp=1.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise.to(device)
        else:
            img = th.randn(*shape, device=device) * temp
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield self.unscale_out_dict(out)
                img = out["sample"]
    # endregion
    
    # region Utils Calculation
    def _vb_terms_bpd(self, model:nn.Module, x_start:th.Tensor, x_t:th.Tensor, t:th.Tensor, clip_denoised=False, model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        if not self.discretized_t0:
            decoder_nll = th.zeros_like(decoder_nll)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            "extra": out["extra"],
        }
        
    def training_losses(
        self, model:nn.Module, x_start:th.Tensor, t:th.Tensor, model_kwargs=None, noise=None
    ) -> Dict[str, th.Tensor]:
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        x_start = self.scale_channels(x_start)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == "kl" or self.loss_type == "rescaled_kl":
            vb_terms = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            terms["loss"] = vb_terms["output"]
            if self.loss_type == "rescaled_kl":
                terms["loss"] *= self.num_timesteps
            extra = vb_terms["extra"]
        elif self.loss_type == "mse" or self.loss_type == "rescaled_mse":
            model_output = model(x_t, t, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output, extra = model_output
            else:
                extra = {}

            if self.model_var_type in [
                "learned",
                "learned_range",
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == "rescaled_mse":
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                "x_prev": self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                "x_start": x_start,
                "epsilon": noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        if "losses" in extra:
            terms.update({k: loss for k, (loss, _scale) in extra["losses"].items()})
            for loss, scale in extra["losses"].values():
                terms["loss"] = terms["loss"] + loss * scale

        return terms
    
    def _prior_bpd(self, x_start:th.Tensor):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model:nn.Module, x_start:th.Tensor, clip_denoised=False, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
    # endregion
    
    
    # region Scaling 
    def scale_channels(self, x: th.Tensor) -> th.Tensor:
        if self.channel_scales is not None:
            x = x * th.from_numpy(self.channel_scales).to(x).reshape(
                [1, -1, *([1] * (len(x.shape) - 2))]
            )
        if self.channel_biases is not None:
            x = x + th.from_numpy(self.channel_biases).to(x).reshape(
                [1, -1, *([1] * (len(x.shape) - 2))]
            )
        return x

    def unscale_channels(self, x: th.Tensor) -> th.Tensor:
        if self.channel_biases is not None:
            x = x - th.from_numpy(self.channel_biases).to(x).reshape(
                [1, -1, *([1] * (len(x.shape) - 2))]
            )
        if self.channel_scales is not None:
            x = x / th.from_numpy(self.channel_scales).to(x).reshape(
                [1, -1, *([1] * (len(x.shape) - 2))]
            )
        return x

    def unscale_out_dict(
        self, out: Dict[str, Union[th.Tensor, Any]]
    ) -> Dict[str, Union[th.Tensor, Any]]:
        return {
            k: (self.unscale_channels(v) if isinstance(v, th.Tensor) else v) for k, v in out.items()
        }
    # endregion