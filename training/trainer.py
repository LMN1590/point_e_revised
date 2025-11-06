import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from typing import Any,Dict
import typing
import os

from diffusers.training_utils import EMAModel

from custom_diffusion.models.transformer.gripper_rep_diffusion_transformer import GripperRepDiffusionTransformer
from custom_diffusion.diffusion.gaussian_diffusion.gaussian_diff_class import GaussianDiffusion

class DiffusionTrainer(LightningModule):
    def __init__(
        self,
        noise_pred_net:GripperRepDiffusionTransformer,
        diffusion:GaussianDiffusion,
        
        ema_power:float = 0.75,
        ema_update_after_step:int = 0,
        
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 0,
        
        num_epochs:int = 1000
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
        
    @property
    def noise_pred_net(self) -> GripperRepDiffusionTransformer:
        return typing.cast(
            GripperRepDiffusionTransformer, self.ema_nets.get_submodule("noise_pred_net")
        )
    
    def training_step(self,gripper_data:Dict[str,torch.Tensor],batch_idx:int):
        """
        Args:
            gripper_data (Dict[str,torch.Tensor]): 2 keys 'grippers' - [B,sample,gripper_dim_mask,finger*segments] and 'object_embeddings' - [B,1,feature_dim,n_ctx]
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
        
        loss_dict = self.diffusion.training_losses(
            model=self.noise_pred_net,
            x_start = grippers,
            t = timesteps,
            model_kwargs = {
                "embeddings": object_encoding
            }
        )
        breakpoint()
        
        
        return None
    def validation_step(self,gripper_data:Dict[str,torch.Tensor],batch_idx:int):
        """
        Args:
            gripper_data (Dict[str,torch.Tensor]): 2 keys 'grippers' - [B,sample,gripper_dim_mask,finger*segments] and 'object_embeddings' - [B,1,feature_dim,n_ctx]
            batch_idx (int): Index of current batch
            
        """
        return None
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.ema_nets.parameters(),lr = self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.num_epochs,eta_min=0.0)
        return [self.optimizer], [self.lr_scheduler]
    
