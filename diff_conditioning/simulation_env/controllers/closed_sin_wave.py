import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List

from .base import Base

class SinWaveClosedLoop(Base):
    '''
    Similar to Open Loop Sine Wave, not the simple kind
    However, in this case the weight and bias is not just learnable parameters, but it is inferenced from a two heads using basic observations.
    '''
    
    def __init__(
        self,
        obs_space,obs_names:List[str],
        hidden_filters: List[int], activation:str,
        n_actuators:int,
        n_sin_waves:int,
        actuation_omega:List[float],
        actuation_strength,
        lr:float,device:str
    ):
        super(SinWaveClosedLoop, self).__init__()

        in_features = 0
        
        for obs_name in obs_names:
            assert obs_name in obs_space.spaces.keys(), f'{obs_name} not in observation space'
            obs_shape = obs_space.spaces[obs_name].shape
            assert len(obs_shape) == 1, f'Only support 1-d observation space, not {obs_shape}'
            in_features += obs_shape[0]
            
        filters = [in_features] + hidden_filters

        net:List[nn.Module] = []
        for i in range(len(filters) - 1):
            net.append(nn.Linear(in_features=filters[i], out_features=filters[i+1]))
            net.append(getattr(nn, activation)())
        self.net = nn.Sequential(*net)

        self.head_weight = nn.Sequential(nn.Linear(filters[-1], n_actuators * n_sin_waves * len(actuation_omega)))
        self.head_bias = nn.Sequential(nn.Linear(filters[-1], n_actuators))

        self.obs_names = obs_names
        self.n_actuators = n_actuators
        self.n_sin_waves = n_sin_waves
        self.actuation_omega = actuation_omega
        self.actuation_strength = actuation_strength
        
        self.device = torch.device(device)
        self.to(device)

        self.optim = optim.Adam(self.parameters(), lr=lr)

    def reset(self):
        self.all_s = []
        self.all_act = []

    def forward(self, s, inp):
        # construct input
        self.all_s.append(s)
        x = []
        for obs_name in self.obs_names:
            x.append(inp[obs_name].float())
        x = torch.cat(x)
        x = x.to(self.device)

        # generate weight and bias
        z = self.net(x)
        weight = self.head_weight(z)
        weight = weight.reshape(self.n_actuators, -1)
        bias = self.head_bias(z)

        # get actuation from sin wave
        time = inp['time'].float()
        x_sinwave = []
        for actuation_omega in self.actuation_omega:
            x_sinwave.append(torch.sin(actuation_omega * time + 2 * torch.pi / self.n_sin_waves * torch.arange(self.n_sin_waves)))
        x_sinwave = torch.cat(x_sinwave)
        x_sinwave = x_sinwave.to(self.device) 
        act = weight @ x_sinwave
        act += bias
        act = torch.tanh(act) * self.actuation_strength
        self.all_act.append(act)

        return act.detach().clone() # NOTE: need clone here otherwise the tensor with the same "identifier"