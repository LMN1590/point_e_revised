import torch

sample = torch.tensor([
    [2,2,2],
    [3,3,3]
])
scaled = torch.tensor([
    0.1,0.2,0.3,0.4
])
offset = torch.tensor([
    [[0.1,0.1,0.1]],
    [[0.2,0.2,0.2]],
    [[0.3,0.3,0.3]],
    [[0.4,0.4,0.4]]
])
# print(sample.shape)
repeated_sample = sample.repeat(4,1,1)
scaled_repeat = scaled[:,None,None]
print(repeated_sample.shape)
print(offset.shape)
print(repeated_sample + offset)