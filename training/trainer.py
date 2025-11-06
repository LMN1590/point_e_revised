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
        
    @property
    def noise_pred_net(self) -> GripperRepDiffusionTransformer:
        return typing.cast(
            GripperRepDiffusionTransformer, self.ema_nets.get_submodule("noise_pred_net")
        )
        
    
    def training_step(self,gripper_embeddings:torch.Tensor,batch_idx):
        print(gripper_embeddings)
    
    # def validation_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
    #     return super().validation_step(*args, **kwargs)
    
