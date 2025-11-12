import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
import numpy as np
from diffusers.training_utils import EMAModel

from typing import Any,Dict,Mapping,Optional
import typing
import os
from tqdm import tqdm
import json

import open3d as o3d

from custom_diffusion.models.transformer.gripper_rep_diffusion_transformer import GripperRepDiffusionTransformer
from custom_diffusion.diffusion.gaussian_diffusion.gaussian_diff_class import GaussianDiffusion
from custom_diffusion.diffusion.diff_utils import _extract_into_tensor

from diff_conditioning.simulation_env.designer.encoded_finger.design_bare import EncodedFingerBare

from .scheduler import warmup_lambda

class DiffusionTrainer(LightningModule):
    def __init__(
        self,
        noise_pred_net:nn.Module,
        diffusion:GaussianDiffusion,
        
        ema_power:float = 0.75,
        ema_update_after_step:int = 0,
        
        learning_rate: float = 1e-4,
        lr_warmup_percentage: int = 0,
        warmup_lr_ratio:float = 0.1,
        
        num_epochs:int = 1000,
        acc_threshold:float = 0.01,
        
        gripper_dim: int = 10,
        max_num_segments: int = 10,
        num_fingers: int = 4,
        pcd_log_dir:str = '',
        
        total_num_steps:int = 100
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["noise_pred_net", "diffusion"])
        self.total_num_steps = total_num_steps
        
        self.ema_nets = nn.ModuleDict({"noise_pred_net": noise_pred_net})
        self.diffusion = diffusion
        
        self.ema = None
        self.ema_power = ema_power
        self.ema_update_after_step = ema_update_after_step
        
        self.learning_rate= learning_rate
        self.lr_warmup_percentage = lr_warmup_percentage
        self.warmup_lr_ratio = warmup_lr_ratio
        
        self.num_epochs = num_epochs
        self.acc_threshold = acc_threshold
        self._compiled_flag = False
        
        self.num_fingers= num_fingers
        self.max_num_segments = max_num_segments
        self.gripper_dim = gripper_dim
        self.total_dim = gripper_dim+1
        self.pcd_log_dir = pcd_log_dir
        with open('diff_conditioning/simulation_env/designer/encoded_finger/config/base_config.json') as f:
            config = json.load(f)
        self.designer = EncodedFingerBare(config,str(self.device))
        
    @property
    def noise_pred_net(self) -> GripperRepDiffusionTransformer:
        return typing.cast(
            GripperRepDiffusionTransformer, self.ema_nets.get_submodule("noise_pred_net")
        )
    
    def setup(self, stage: Optional[str] = None):
        """Called after Lightning moves model to the right device."""
        if os.environ.get("TORCH_COMPILE", "0") != "0" and not self._compiled_flag:
            # compile the module in-place and replace module
            compiled = torch.compile(self.noise_pred_net, mode="max-autotune")
            self.ema_nets["noise_pred_net"] = compiled
            self._compiled_flag = True

        # 2) Create EMA after device placement
        if self.ema is None:
            self.ema = EMAModel(
                parameters=self.ema_nets.parameters(),
                power=self.ema_power,
                update_after_step=self.ema_update_after_step,
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
        sample_weights_norm = gripper_data['weights'] / gripper_data['weights'].sum(1)[:,None]
        sample_weights = sample_weights_norm.flatten(0,1) # [B*sample]
        
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
        mean_loss_dict = {k: (v * sample_weights / B).sum() for k,v in loss_dict.items() if "loss" in k}
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
        x_start = self.diffusion.scale_channels(grippers)
        
        object_encoding = gripper_data['object_embedding'].flatten(0,1) # [B*sample,obj_dim,n_ctx_obj]
        model_kwargs = {
            "embeddings": object_encoding
        }
        
        sample_weights_norm = gripper_data['weights'] / gripper_data['weights'].sum(1)[:,None]
        sample_weights = sample_weights_norm.flatten(0,1).detach() # [B*sample]
        
        noise = torch.randn(x_start.shape,device=self.device) # [B*sample_size, gripper_dim_mask, finger*segments ~ n_ctx]
        t_end = (self.diffusion.num_timesteps-1) * torch.ones((B*S,),dtype=torch.int64,device=self.device)
        x_t = self.diffusion.q_sample(x_start,t_end,noise)
        
        noise_pred_loss = torch.tensor(0,device=self.device,dtype=torch.float)
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        for i in indices:
            t = torch.full((B*S,),i,dtype=torch.int64,device=self.device)
            with torch.no_grad():
                model_output = self.noise_pred_net(x_t,t,**model_kwargs)
                
                # Calculate Variance
                if self.diffusion.model_var_type in ['learned','learned_range']:
                    assert model_output.shape == (B*S,C*2,T)
                    model_pred,model_var_values = torch.split(model_output,C,dim=1)
                    if self.diffusion.model_var_type == 'learned_range':
                        min_log = _extract_into_tensor(self.diffusion.posterior_log_variance_clipped, t, x_t.shape)
                        max_log = _extract_into_tensor(np.log(self.diffusion.betas), t, x_t.shape)
                        # The model_var_values is [-1, 1] for [min_var, max_var].
                        
                        model_var_values = model_var_values.clamp(-1,1)
                        frac = (model_var_values + 1) / 2
                        model_log_variance = frac * max_log + (1 - frac) * min_log
                    else:
                        raise NotImplementedError(f"The current model variance type {self.diffusion.model_var_type} has not been implemented")
                else:
                    raise NotImplementedError(f"The current model variance type {self.diffusion.model_var_type} has not been implemented")
                
                # Calculate Epsilon
                if self.diffusion.model_mean_type == 'epsilon':
                    pred_xstart = self.diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_pred)
                    model_mean, _, _ = self.diffusion.q_posterior_mean_variance(
                        x_start=pred_xstart, 
                        x_t=x_t, t=t
                    )
                else:
                    raise NotImplementedError(f"Not implemented during validation: {self.diffusion.model_mean_type}")
                
                # Loss Calculation
                target = {
                    "x_prev": self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                    "x_start": x_start,
                    "epsilon": self.diffusion._predict_eps_from_xstart(x_t=x_t,t=t,pred_xstart=x_start),
                }[self.diffusion.model_mean_type]
                assert model_pred.shape == target.shape == x_start.shape
                
                cur_mse_loss = ((model_pred - target)**2).flatten(1).mean(1) * sample_weights / B # [B*sample]
                noise_pred_loss += cur_mse_loss.sum()
                
                # Reconstruct previous x_prev
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * torch.randn_like(x_t)
                x_t = x_t.clamp(-1,1)

        indiv_final_sample_loss = ((x_t - x_start) ** 2).flatten(1).mean(1)  # [B*sample_size]
        min_id = torch.argmin(indiv_final_sample_loss)
        self._reconstruct_gripper(x_start[min_id],f'groundtruth_epoch_{self.current_epoch}_loss_{indiv_final_sample_loss[min_id].item()}.ply')
        self._reconstruct_gripper(x_t[min_id],f'pred_epoch_{self.current_epoch}_loss_{indiv_final_sample_loss[min_id].item()}.ply')
        
        final_sample_loss = indiv_final_sample_loss * sample_weights / B
        accuracy = torch.mean(torch.abs(x_t - x_start) < self.acc_threshold, dtype=torch.float)
        self.log_dict(
            {
                "val/noise_pred_loss": (noise_pred_loss/self.diffusion.num_timesteps).item(),
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
        warmup_iter = int(np.round(self.lr_warmup_percentage * self.num_epochs))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=warmup_lambda(
                warmup_steps=warmup_iter,
                min_lr_ratio=self.warmup_min_lr_ratio
            )
        )
        
        
        
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.num_epochs,eta_min=0.0)
        return [self.optimizer], [self.lr_scheduler]
    

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        assert self.ema is not None
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
        assert self.ema is not None
        retval = super().load_state_dict(state_dict, strict=False)
        self.ema.load_state_dict(state_dict["ema_model"])
        return retval

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        assert self.ema is not None
        print('on save checkpoint')
        checkpoint["state_dict"]["ema_model"] = self.ema.state_dict()
        super().on_save_checkpoint(checkpoint)
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # reference: https://lightning.ai/docs/pytorch/2.0.9/debug/debugging_intermediate.html#look-out-for-exploding-gradients
        norms = grad_norm(self.noise_pred_net, norm_type=2)
        self.log_dict(norms)
        
    def _reconstruct_gripper(self,tensor:torch.Tensor,filename:str):
        pred_xstart = self.diffusion.unscale_channels(tensor) # [B,C,num_finger*num_segments]
        gripper_emb = pred_xstart.permute(0,2,1).reshape(1,self.num_fingers,self.max_num_segments,self.total_dim) # [B,finger,segments,C]
        
        ctrl_tensors, end_masks = gripper_emb[:,:,:,:-1], gripper_emb[:,:,:,-1]
        #TODO: This is because current implementation do not support suction
        modded_ctrl_tensors = torch.cat([ctrl_tensors,torch.zeros(*ctrl_tensors.shape[:-1],1,device=ctrl_tensors.device,dtype = ctrl_tensors.dtype)],dim=-1)
        
        self.designer.reset()
        pts = self.designer._create_representation_from_tensor(modded_ctrl_tensors[0],end_masks[0])
        points_np = pts.cpu().numpy()

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.paint_uniform_color([0.1, 0.7, 0.9])
        o3d.io.write_point_cloud(os.path.join(self.pcd_log_dir,f"{filename}.ply"), pcd)