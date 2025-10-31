from .base import Base
import torch

from softzoo.envs.base_env import BaseEnv
from ..designer.encoded_finger.base_config import Config

class CustomFingerRepController(Base):
    def __init__(
        self, 
        base_config:Config,
        env:BaseEnv,
        n_actuators:int,
        device:str,
        active:bool,**kwargs
    ):
        super(CustomFingerRepController, self).__init__()
        self.base_config = base_config
        self.env = env
        self.n_actuators = n_actuators
        self.active = active
        self.device = torch.device(device)
        self.to(device)

    def forward(self, s, inp, ctrl_tensor:torch.Tensor):
        self.all_s.append(s)
        
        ctrl_tensor = torch.sigmoid(ctrl_tensor.to(self.device))
        strength_tensor = ctrl_tensor[:,:,9].flatten() # (num_finger*num_segment)
        strength_scaled = strength_tensor * (
            self.base_config['segment_config']['actuation_strength_range'][1] - self.base_config['segment_config']['actuation_strength_range'][0]
        ) + self.base_config['segment_config']['actuation_strength_range'][0] # (B,)
        strength_by_lbls = strength_scaled.repeat_interleave(2)
        full_strength = torch.concat([
            torch.tensor([0.]).to(self.device),
            strength_by_lbls
        ]) # (1+num_finger*num_segment*2)
        
        inp = inp['time'].float()
        
        act_scale = torch.zeros((self.n_actuators,), requires_grad=False)
        act_scale[:min(full_strength.shape[0],self.n_actuators)] += full_strength[:min(full_strength.shape[0],self.n_actuators)]
        act = torch.zeros((self.n_actuators,), requires_grad=False) # (1+num_finger*num_segment*2)
        if self.active:
            if s//16<25:
                act[::2] = -1.0
                act[1::2] = 1.0
            else:   
                act[::2] = 1.0
                act[1::2] = -1.0
        act = act * act_scale
        self.all_act.append(act)
        
        return act.detach().clone()

    def update(self, grad, retain_graph=False):
        pass
    
    def reset(self):
        self.all_s = []
        self.all_act = []

    def backward(self,grad,retain_graph=False):
        all_act = torch.stack(self.all_act)
        grad = torch.stack(grad)
        grad = grad.to(all_act) # make sure they are of the same type

        all_act.backward(gradient=grad, retain_graph=retain_graph)