import torch

sample = torch.randn(3,2,10)
print(sample.flatten(1).mean(1).shape)