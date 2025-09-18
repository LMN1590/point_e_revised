from .base import Base
import torch

from softzoo.envs.base_env import BaseEnv

class AllOn(Base):
    def __init__(
        self, 
        env:BaseEnv,
        n_actuators:int,
        actuation_strength:float,
        device:str,
        active:bool,**kwargs
    ):
        super(AllOn, self).__init__()
        self.env = env
        self.n_actuators = n_actuators
        self.actuation_strength = actuation_strength
        self.active = active
        self.device = torch.device(device)
        self.to(device)

    def forward(self, s, inp):
        inp = inp['time'].float()
        
        act = torch.zeros((self.n_actuators,), requires_grad=False)
        if self.active:
            if s//16<90:
                act[::2] = -1.0
                act[1::2] = 1.0
            else:   
                act[::2] = 1.0
                act[1::2] = -1.0
        act = act * self.actuation_strength
        return act.detach().clone()

    def update(self, grad, retain_graph=False):
        pass
    def reset(self):
        pass
