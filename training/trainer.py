import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from typing import Any
import typing
import os

from diffusers.training_utils import EMAModel

from custom_diffusion.models.transformer.gripper_rep_diffusion_transformer import GripperRepDiffusionTransformer
from custom_diffusion.diffusion.gaussian_diffusion.gaussian_diff_class import GaussianDiffusion

class DiffusionTrainer(LightningModule):
    def __init__(
        self,
        noise_pred_net:GripperRepDiffusionTransformer,
        diffusion_scheduler:GaussianDiffusion,
        
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
        
        self.learning_rate= learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        
        self.num_epochs = num_epochs
        
    @property
    def noise_pred_net(self) -> GripperRepDiffusionTransformer:
        return typing.cast(
            GripperRepDiffusionTransformer, self.ema_nets.get_submodule("noise_pred_net")
        )
        
    
    def training_step(self,gripper_embeddings:torch.Tensor,batch_idx):
        return None
    def validation_step(self,gripper_embeddings:torch.Tensor,batch_idx):
        return None
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.ema_nets.parameters(),lr = self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.num_epochs,eta_min=0.0)
        return [self.optimizer], [self.lr_scheduler]
    
