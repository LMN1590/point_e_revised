import torch

start_idx,end_idx = 128,0

x = torch.tensor([2,3,4,129,230])

res = torch.logical_and(x>=end_idx,x<=start_idx)
print(res)