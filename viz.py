import torch

sample = torch.tensor([
    [1.,1.,1.],
    [2.,2.,2.],
    [3.,3.,3.]
])
print(sample.shape)
rolled_sample = torch.roll(sample,shifts=1,dims=0)
rolled_sample[0] = torch.zeros_like(rolled_sample[0])
print(rolled_sample)