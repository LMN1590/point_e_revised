import numpy as np

from typing import Sequence, Optional

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
        model_mean_type: str,
        model_var_type: str,
        loss_type: str,
        discretized_t0: bool = False,
        channel_scales: Optional[np.ndarray] = None,
        channel_biases: Optional[np.ndarray] = None,
    ):
        self.model_mean_type = model_mean_type