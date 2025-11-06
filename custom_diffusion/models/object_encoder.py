import torch
import torch.nn as nn 

from typing import List
from typing_extensions import TypedDict

class ObjectConfig(TypedDict):
    name:str
    quat:List[float]
    scale:float
    

class EncoderPlaceholder:
    def __init__(
        self
    ):
        super().__init__()
        self.feature_dim = 128
        self.n_ctx = 100
        
        # self.encoder = nn.Linear(2,2)
    
    def encode(self,objects:List[ObjectConfig],batch_size:int):
        # TODO: The transformation and point-cloud-rization of the object should be carried out here, an additional cache to save the embeddings would be great as well. TBD
        return torch.zeros((batch_size,self.feature_dim,self.n_ctx))