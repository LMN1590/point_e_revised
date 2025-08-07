from .base import Base
import torch

from softzoo.envs.base_env import BaseEnv

class AllOn(Base):
    def __init__(
        self, 
        env:BaseEnv,
        n_actuators:int,
        actuation_strength:float,
        device:str,**kwargs
    ):
        super(AllOn, self).__init__()
        self.env = env
        self.n_actuators = n_actuators
        self.actuation_strength = actuation_strength
        self.device = torch.device(device)
        self.to(device)

    def forward(self, s, inp):
        inp = inp['time'].float()
        
        act = torch.zeros((self.n_actuators,), requires_grad=False)
        # print(inp)
        if inp < 0.75:
            act[::2] = 1.0
            act[1::2] = -1.0
        act = act * self.actuation_strength
        return act.detach().clone()

    def update(self, grad, retain_graph=False):
        pass
    def reset(self):
        pass
