import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import numpy as np

from typing import Any,Dict,Mapping
import typing
import os

from diffusers.training_utils import EMAModel

from custom_diffusion.models.transformer.gripper_rep_diffusion_transformer import GripperRepDiffusionTransformer
from custom_diffusion.diffusion.gaussian_diffusion.gaussian_diff_class import GaussianDiffusion
from custom_diffusion.diffusion.diff_utils import _extract_into_tensor

class DiffusionTrainer(LightningModule):
    def __init__(
        self,
        noise_pred_net:nn.Module,
        diffusion:GaussianDiffusion,
        
        ema_power:float = 0.75,
        ema_update_after_step:int = 0,
        
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 0,
        
        num_epochs:int = 1000,
        
        acc_threshold:float = 0.01
    ):
        super().__init__()
        if os.environ.get("TORCH_COMPILE", "0") == "0":
            self.ema_nets = nn.ModuleDict(
                {
                    "noise_pred_net": noise_pred_net,
                }
            )
        else:
            # cache text features before compiling
            self.ema_nets = nn.ModuleDict(
                {
                    "noise_pred_net": torch.compile(noise_pred_net, mode="max-autotune"),
                }  # type: ignore
            )
        self.ema = EMAModel(
            parameters=self.ema_nets.parameters(),
            power=ema_power,
            update_after_step=ema_update_after_step,
        )
        self.diffusion = diffusion
        
        self.learning_rate= learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        
        self.num_epochs = num_epochs
        
        self.acc_threshold = acc_threshold
        
    @property
    def noise_pred_net(self) -> GripperRepDiffusionTransformer:
        return typing.cast(
            GripperRepDiffusionTransformer, self.ema_nets.get_submodule("noise_pred_net")
        )
    
    def training_step(self,gripper_data:Dict[str,torch.Tensor],batch_idx:int):
        """
        Args:
            gripper_data (Dict[str,torch.Tensor]): 3 keys 
                'grippers' - [B,sample,gripper_dim_mask,finger*segments]
                'object_embeddings' - [B,sample,feature_dim,n_ctx]
                'weights' - [B,sample]
            batch_idx (int): Index of current batch
        """
        B,S = gripper_data['grippers'].shape[:2]
        grippers = gripper_data['grippers'].flatten(0,1) # [B*sample_size,gripper_dim_mask, finger*segments ~ n_ctx]
        object_encoding = gripper_data['object_embedding'].flatten(0,1) # [B*sample,obj_dim,n_ctx_obj]
        
        timesteps = torch.randint(
            0,
            self.diffusion.num_timesteps,  # type: ignore
            (B * S,),
            device=self.device,
        ).long()
        
        loss_dict:Dict[str,torch.Tensor] = self.diffusion.training_losses(
            model=self.noise_pred_net,
            x_start = grippers,
            t = timesteps,
            model_kwargs = {
                "embeddings": object_encoding
            }
        )
        mean_loss_dict = {k: v.mean() for k,v in loss_dict.items() if "loss" in k}
        self.log_dict(
            {f"train/mean_{k}": v for k,v in mean_loss_dict.items()},
            sync_dist=True,
            on_step = True,
            on_epoch = True,
            # prog_bar=True
        )
        return mean_loss_dict['total_loss']
    
    def validation_step(self,gripper_data:Dict[str,torch.Tensor],batch_idx:int):
        """
        Args:
            gripper_data (Dict[str,torch.Tensor]): 3 keys 
                'grippers' - [B,sample,gripper_dim_mask,finger*segments]
                'object_embeddings' - [B,sample,feature_dim,n_ctx]
                'weights' - [B,sample]
            batch_idx (int): Index of current batch
        """
        B,S,C,T = gripper_data['grippers'].shape
        grippers = gripper_data['grippers'].flatten(0,1) # [B*sample_size, gripper_dim_mask, finger*segments ~ n_ctx]
        scaled_grippers = self.diffusion.scale_channels(grippers)
        
        object_encoding = gripper_data['object_embedding'].flatten(0,1) # [B*sample,obj_dim,n_ctx_obj]
        model_kwargs = {
            "embeddings": object_encoding
        }
        
        sample_weights_norm = gripper_data['weights'] / gripper_data['weights'].sum(1)[:,None]
        sample_weights = sample_weights_norm.flatten(0,1) # [B*sample]
        
        noise = torch.randn(scaled_grippers.shape,device=self.device) # [B*sample_size, gripper_dim_mask, finger*segments ~ n_ctx]
        original_timesteps = (self.diffusion.num_timesteps-1) * torch.ones((B*S,),dtype=torch.int64,device=self.device)
        sample = self.diffusion.q_sample(scaled_grippers,original_timesteps,noise)
        
        noise_pred_loss = 0.0
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        for t in indices:
            timesteps = t * torch.ones((B*S,),dtype=torch.int64,device=self.device)
            with torch.no_grad():
                model_output = self.noise_pred_net(sample,timesteps,**model_kwargs)
                if self.diffusion.model_var_type in ['learned','learned_range']:
                    assert model_output.shape == (B*S,C*2,T)
                    model_mean,model_var_values = torch.split(model_output,C,dim=1)
                    if self.diffusion.model_var_type == 'learned_range':
                        min_log = _extract_into_tensor(self.diffusion.posterior_log_variance_clipped, timesteps, sample.shape)
                        max_log = _extract_into_tensor(np.log(self.diffusion.betas), timesteps, sample.shape)
                        # The model_var_values is [-1, 1] for [min_var, max_var].
                        
                        model_var_values = model_var_values.clamp(-1,1)
                        frac = (model_var_values + 1) / 2
                        model_log_variance = frac * max_log + (1 - frac) * min_log
                        
                        nonzero_mask = (
                            (timesteps != 0).float().view(-1, *([1] * (len(sample.shape) - 1)))
                        )  # no noise when t == 0
                        
                        noise = torch.randn_like(sample)
                        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
                        sample = sample.clamp(-1,1)
                    else:
                        raise NotImplementedError(f"The current model variance type {self.diffusion.model_var_type} has not been implemented")
                else:
                    raise NotImplementedError(f"The current model variance type {self.diffusion.model_var_type} has not been implemented")
                cur_mse_loss = ((model_mean - noise)**2).flatten(1).mean(1) * sample_weights / B # [B*sample]
                noise_pred_loss += cur_mse_loss.sum().item()

        noise_pred_loss /= self.diffusion.num_timesteps
        final_sample_loss = ((sample - scaled_grippers) ** 2).flatten(1).mean(1) * sample_weights / B
        accuracy = torch.mean(torch.abs(sample - scaled_grippers) < self.acc_threshold, dtype=torch.float)
        self.log_dict(
            {
                "val/noise_pred_loss": noise_pred_loss,
                "val/final_sample_loss": final_sample_loss.sum().item(),
                "val/accuracy": accuracy,
            },
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar = True
        )
        
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.ema_nets.parameters(),lr = self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.num_epochs,eta_min=0.0)
        return [self.optimizer], [self.lr_scheduler]
    

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        try:
            self.ema.step(self.ema_nets.parameters())
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                self.ema.to(self.device)
                self.ema.step(self.ema_nets.parameters())
            else:
                raise e
        self.log(
            "train/ema_decay",
            self.ema.decay,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar = True
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if os.environ.get("TORCH_COMPILE", "0") == "0":
            # need to handle torch compile, for instance:
            # noise_pred_net._orig_mod.final_conv.1.bias
            # noise_pred_net.final_conv.1.bias
            checkpoint["state_dict"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
            checkpoint["state_dict"]["ema_model"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"]["ema_model"].items()
            }
        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        retval = super().load_state_dict(state_dict, strict=False)
        self.ema.load_state_dict(state_dict["ema_model"])
        return retval

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        print('on save checkpoint')
        checkpoint["state_dict"]["ema_model"] = self.ema.state_dict()
        super().on_save_checkpoint(checkpoint)