import numpy as np
import torch
import os


base_path = 'data/grippers/'
sigmoided_path = 'data/grippers_norm'
os.makedirs(sigmoided_path,exist_ok=True)

for index in range(1000):  # 0 to 999
    sample_gripper = os.path.join(base_path,f'gripper_nf4_id{index}.npz')
    content = np.load(sample_gripper)

    end_prob = content['end_prob']  # shape [4, 10]
    breakpoint()
    print(end_prob)
    break
