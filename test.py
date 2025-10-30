import torch

fingers_pos = torch.randn(4,10,3)

dist = torch.cdist(fingers_pos[1], fingers_pos[0])  # (num_points,pts,3)
dist_threshold = 0.01
summed = torch.sum(torch.clamp(dist_threshold - dist, min=0.0))