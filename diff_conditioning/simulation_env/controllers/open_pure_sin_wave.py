import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List

from .base import Base

class PureSinWaveOpenLoop(Base):
    '''
    Each actuator is assigned a sine wave that determines its signal
    omega = 2pi*omega*mul
    act = amp * sin(omega*t + psi) * strength
    '''
    
    def __init__(
        self,
        n_actuators:int, actuation_strength:float,
        omega_mul:List[float],
        lr:float, device:str
    ):
        super(PureSinWaveOpenLoop, self).__init__()

        self.n_actuators = n_actuators
        self.actuation_strength = actuation_strength
        self.omega_mul = omega_mul
        self.device = torch.device(device)

        self.amp = nn.Parameter(torch.ones((self.n_actuators,), requires_grad=True))
        self.omega = nn.Parameter(torch.ones((self.n_actuators,), requires_grad=True))
        self.psi = nn.Parameter(torch.zeros((self.n_actuators,), requires_grad=True))

        self.to(device)
        self.omega.data.normal_(0.0, 1.0)

        self.optim = optim.Adam([self.amp, self.omega, self.psi], lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        inp = inp['time'].float()
        self.all_s.append(s)
        omega = 2 * torch.pi * self.omega * self.omega_mul
        act = self.amp * torch.sin(omega * inp + self.psi) * self.actuation_strength
        self.all_act.append(act)

        return act.detach().clone() # NOTE: need clone here otherwise the tensor with the same "identifier"
