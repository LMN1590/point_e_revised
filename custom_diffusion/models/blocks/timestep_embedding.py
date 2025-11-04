import math

import torch


def timestep_embedding(timesteps:torch.Tensor, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # [N,dim]
    return embedding

def finger_segment_geo_embedding(
    num_fingers:int, max_num_segments:int,total_dim:int
):
    '''
    Generating sinusoidal embeddings for geometrical for fingers and segments respectively.
    
    :return: an [num_fingers,max_num_segments,total_dim] tensor of positional embeddings, averaged between the per-finger and per-segment embeddings.
    '''
    segment_ids = torch.arange(max_num_segments)
    segment_emb = timestep_embedding(segment_ids, total_dim) # [max_num_segments,total_dim]
    
    # Fixed sinusoidal (or constant) for fingers
    finger_ids = torch.arange(num_fingers)
    finger_emb = timestep_embedding(finger_ids, total_dim) # # [num_fingers,total_dim]
    
    # Combine them: [num_fingers, max_segments, dim]
    # Each finger shares the same segment structure but distinct offset
    pos_enc = (finger_emb[:, None, :] + segment_emb[None, :, :]) * 0.5
    
    # Flatten into (num_fingers * max_segments,dim)
    pos_enc = pos_enc.reshape(num_fingers * max_num_segments, total_dim)
    return pos_enc
    
    