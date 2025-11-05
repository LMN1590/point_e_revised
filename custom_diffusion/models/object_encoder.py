import torch
import torch.nn as nn 

from typing import List

class EncoderPlaceholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 128
        self.n_ctx = 100
        
        # self.encoder = nn.Linear(2,2)
    
    def forward(self,objects:List[str]):
        return torch.zeros((1,self.feature_dim,self.n_ctx))