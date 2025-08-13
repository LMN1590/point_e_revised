import numpy as np
import torch

loaded_pcd = np.load('sample_generated_pcd.npz')
x = torch.from_numpy(loaded_pcd['coords']).detach()
x=torch.stack([x,x],dim=0)
print(x.shape)