import torch
import torch.nn as nn

from typing import Sequence, Dict, Any, Iterator, Callable,Tuple, List, Union, Optional,Literal

from .gaussian_diffusion import GaussianDiffusion,SpacedDiffusion

CONDITIONING_KEY = Literal['objects','embeddings']

class GripperSampler:
    """
    An adapted sampler for generating gripper in a diffusion model, modified to produce conditional or unconditional samples.
    """
    
    def __init__(
        self,
        device: torch.device,
        models: Sequence[nn.Module],
        diffusions: List[Union[GaussianDiffusion,SpacedDiffusion]],
        model_kwargs_key_filter: Sequence[CONDITIONING_KEY] = ("embeddings",),
        guidance_scale: Sequence[float] = (3.0,),
        clip_denoised: bool = True,
        sampling_mode:Literal['ddpm','ddim'] = 'ddpm',
        
        gripper_dim: int = 10,
        max_num_segments: int = 10,
        num_fingers: int = 4
    ):
        n = len(models)
        assert n>0
    
        if len(model_kwargs_key_filter)==0: model_kwargs_key_filter = ["embeddings",] * n
        
        self.device = device
        self.model_kwargs_key_filter = model_kwargs_key_filter
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.sampling_mode = sampling_mode

        self.models = models
        self.diffusions = diffusions
        
        self.gripper_dim = gripper_dim
        self.total_dim = gripper_dim + 1
        self.max_num_segments = max_num_segments
        self.num_fingers = num_fingers
        
        for model in self.models:
            if hasattr(model, "_init_fingers_topo"): model._init_fingers_topo(
                gripper_dim = self.gripper_dim,
                max_num_segments = self.max_num_segments,
                num_fingers = self.num_fingers
            )
        
        self._validate_params()
    
    @property
    def num_stages(self) -> int:
        return len(self.models)
    
    def _validate_params(self):
        n = self.num_stages
        assert len(self.diffusions) == n
        assert len(self.model_kwargs_key_filter) == n
        assert len(self.guidance_scale) == n
        assert self.sampling_mode in ['ddpm','ddim']
    # endregion
        
    # region Sampling
    ######################################
    ############## Sampling ##############
    ######################################
    def sample_batch(
        self, 
        batch_size: int, model_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        samples = None
        for x in self.sample_batch_progressive(batch_size, model_kwargs):
            samples = x
        return samples

    def sample_batch_progressive(
        self,
        batch_size: int, model_kwargs: Dict[str, Any],
        pre_noise:List[Optional[torch.Tensor]]=[None],
        cond_fn_lst:List[Optional[Callable[...,torch.Tensor]]] = [None]
    )->Iterator[torch.Tensor]:
        assert len(pre_noise) == self.num_stages
        samples = None
        for (
            model, diffusion,
            stage_guidance_scale,
            stage_key_filter,
            noise, cond_fn
        ) in zip(
            self.models, self.diffusions,
            self.guidance_scale,
            self.model_kwargs_key_filter,
            pre_noise,cond_fn_lst
        ):
            # breakpoint()
            stage_model_kwargs = model_kwargs.copy()
            if stage_key_filter != "*":
                # Filter the model kwargs to only include keys that are in the filter
                use_keys = set(stage_key_filter.split(","))
                stage_model_kwargs = {k: v for k, v in stage_model_kwargs.items() if k in use_keys}
            if hasattr(model, "cached_model_kwargs") and "embeddings" not in stage_model_kwargs:
                # Overwrite the model kwargs with the processed ones
                # If key "objects" are provided, we need to use cached_model_kwargs to process them
                stage_model_kwargs = model.cached_model_kwargs(batch_size, stage_model_kwargs)
                
            sample_shape = (batch_size, self.total_dim, self.num_fingers*self.max_num_segments) # (batch, dims, segments_fingers)
            
            if stage_guidance_scale != 1 and stage_guidance_scale != 0:
                for k, v in stage_model_kwargs.copy().items():
                    stage_model_kwargs[k] = torch.cat([v, torch.zeros_like(v)], dim=0) # (dim)
                    
            internal_batch_size = batch_size
            if stage_guidance_scale:
                model = self._uncond_guide_model(model, stage_guidance_scale)
                internal_batch_size *= 2
            sample_loop_progressive = diffusion.p_mcmc_sample_loop_progressive if self.sampling_mode =='ddpm' else diffusion.ddim_mcmc_sample_loop_progressive
            # If unconditional guidance is used, double the batch size: (2B, dims, segments_fingers)
            
            samples_it = sample_loop_progressive(
                model,
                shape=(internal_batch_size, *sample_shape[1:]),
                model_kwargs=stage_model_kwargs,
                device=self.device,
                clip_denoised=self.clip_denoised,
                noise = noise,
                cond_fn = cond_fn
            )
                
                
            for x in samples_it:
                samples = x['pred_xstart'][:batch_size]
                if "low_res" in stage_model_kwargs:
                    samples = torch.cat([stage_model_kwargs["low_res"][:len(samples)], samples], dim=-1)
                yield samples
            
    
    def _uncond_guide_model(
        self, model: Callable[..., torch.Tensor], scale: float
    ) -> Callable[..., torch.Tensor]:
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
            half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        return model_fn
    # endregion