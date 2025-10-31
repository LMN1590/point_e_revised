import torch
import numpy as np

import os
from tqdm import tqdm

def generate_gripper(
    num_finger:int,variance_scale:float,max_segment_count:int,hidden_dim:int,
    num_grippers:int,gripper_dir:str    
):
    os.makedirs(gripper_dir,exist_ok=True)
    
    gripper_emb = torch.randn(num_grippers, num_finger, max_segment_count, hidden_dim) * variance_scale
    gripper_emb[:,:,:,6] *= (0.1/variance_scale)
    gripper_emb[:,:,:,-1] += 0.75
    end_prob = torch.randn(num_grippers, num_finger, max_segment_count)
    
    for i in tqdm(range(num_grippers)):
        gripper_path = os.path.join(gripper_dir,f'gripper_nf{num_finger}_id{i}.npz')
        np.savez_compressed(
            gripper_path,
            gripper_emb = gripper_emb[i].cpu().numpy(),
            end_prob = end_prob[i].cpu().numpy()
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_finger',type=int,default=4, help="Number of fingers")
    parser.add_argument("--variance_scale",type=float,default=1.0,help="Scale of variance")
    parser.add_argument("--max_segment_count",type=int,default=10,help="Maximum segment count per finger")
    parser.add_argument("--hidden_dim",type=int,default=10,help="Hidden dimension size")
    
    parser.add_argument("--num_grippers",type=int,default=100,help="Number of grippers")
    parser.add_argument("--gripper_dir",type=str,default="data/grippers",help="Directory to save grippers")
    
    args = parser.parse_args()
    generate_gripper(
        num_finger=args.num_finger,
        variance_scale=args.variance_scale,
        max_segment_count=args.max_segment_count,
        hidden_dim=args.hidden_dim,
        num_grippers=args.num_grippers,
        gripper_dir=args.gripper_dir
    )