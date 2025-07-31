import torch.nn as nn
import torch

import math

from ..util import init_linear
from ..checkpoint import checkpoint
from .mlp import MLP

class MultiheadAttention(nn.Module):
    def __init__(
        self, *,
        device:torch.device, dtype:torch.dtype,
        heads:int, n_ctx:int,
        width:int, init_scale:float
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(
            width, width*3,
            device = device, dtype = dtype
        )
        self.c_proj = nn.Linear(
            width,width,
            device=device,dtype=dtype
        )
        self.attention = QKVMultiheadAttention(
            device = device, dtype = dtype,
            heads = heads, n_ctx = n_ctx
        )
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)
        
    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x
        
class QKVMultiheadAttention(nn.Module):
    """
    Multihead attention module that computes Q, K, V matrices
    """
    def __init__(
        self, *, 
        device:torch.device, dtype: torch.dtype,
        heads:int, n_ctx:int
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx
        
    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1/math.sqrt(math.sqrt(attn_ch))
        
        qkv = qkv.view(bs,n_ctx,self.heads,-1)
        q,k,v = torch.split(qkv, attn_ch, dim=-1)
        
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, *,
        device:torch.device, dtype:torch.dtype,
        heads:int, n_ctx:int,
        width:int, init_scale:float = 1.0
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            device = device, dtype = dtype,
            heads = heads, n_ctx = n_ctx,
            width = width, init_scale = init_scale
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x