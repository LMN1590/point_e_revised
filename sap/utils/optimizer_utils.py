import torch

from .schedule_utils import StepLearningRateSchedule

def update_optimizer(
    inputs:torch.Tensor,
    schedule:StepLearningRateSchedule,
    epoch:int
):
    return torch.optim.Adam(
        [inputs],
        lr=schedule.get_learning_rate(epoch)
    )