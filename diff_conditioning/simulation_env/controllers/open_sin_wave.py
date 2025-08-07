import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List

from .base import Base

class SinWaveOpenLoop(Base):
    '''
    Information from each sine waves plays a part in the actuation signal.
    
    Input: time t
    x(t) = 2pi * sine_index/sine_count + omega*t   
    x'(t) = A*x(t) + B
    tanh(x'(t)) * actuation_strength
    '''
    
    def __init__(
        self,
        n_actuators:int,
        n_sin_waves:int,
        actuation_omega:List[float],
        actuation_strength:float,
        lr:float,
        device:str
    ):
        super(SinWaveOpenLoop, self).__init__()

        self.n_actuators = n_actuators
        self.n_sin_waves = n_sin_waves
        self.actuation_omega = actuation_omega
        self.actuation_strength = actuation_strength
        self.device = torch.device(device)

        self.weight = nn.Parameter(torch.zeros((self.n_actuators, self.n_sin_waves * len(self.actuation_omega)), requires_grad=True))
        self.bias = nn.Parameter(torch.zeros((self.n_actuators,), requires_grad=True))

        self.to(device)
        self.weight.data.normal_(0.0, 0.3)

        self.optim = optim.Adam([self.weight, self.bias], lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        inp = inp['time'].float()
        self.all_s.append(s)

        x = []
        for actuation_omega in self.actuation_omega:
            x.append(torch.sin(actuation_omega * inp + 2 * torch.pi / self.n_sin_waves * torch.arange(self.n_sin_waves)))
        x = torch.cat(x)
        x = x.to(self.device) 
        act = self.weight @ x
        act += self.bias
        act = torch.tanh(act) * self.actuation_strength
        self.all_act.append(act)

        return act.detach().clone() # NOTE: need clone here otherwise the tensor with the same "identifier"

    def update(self, grad, retain_graph=False):
        all_act = torch.stack(self.all_act)
        grad = torch.stack(grad)
        grad = grad.to(all_act) # make sure they are of the same type

        self.optim.zero_grad()
        all_act.backward(gradient=grad, retain_graph=retain_graph)
        self.optim.step()