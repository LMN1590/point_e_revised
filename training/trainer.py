from typing import Any
from pytorch_lightning import LightningModule

class DiffusionTrainer(LightningModule):
    def __init__(self):
        pass
    def training_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
        return super().training_step(*args, **kwargs)
    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)