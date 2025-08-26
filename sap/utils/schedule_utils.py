import numpy as np
import torch

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass
class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor, final=1e-6):
        self.initial = float(initial)
        self.interval = interval
        self.factor = factor
        self.final = float(final)

    def get_learning_rate(self, epoch):
        lr = np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)
        return lr if lr > self.final else self.final


def adjust_learning_rate(lr_schedule:StepLearningRateSchedule, optimizer:torch.optim.Optimizer, epoch:int):
    for _, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedule.get_learning_rate(epoch)