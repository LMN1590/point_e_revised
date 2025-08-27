import torch

def gaussian_kernel(surface_pc:torch.Tensor, dense_pc:torch.Tensor,alpha:float = 20.):
    dists = torch.cdist(surface_pc, dense_pc, p=2)**2
    K = torch.exp(-alpha * dists)
    return K / (K.sum(dim=1, keepdim=True) + 1e-9)