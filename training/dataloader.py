import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

from typing import Literal

import math
import pandas as pd
from typing import List
import os

from custom_diffusion.models.object_encoder import EncoderPlaceholder,ObjectConfig

class GripperDataset(Dataset):
    """
    Custom dataset for diffusion training.
    
    Input CSV columns:
        - obj_id
        - obj_quat (e.g. quaternion as string or 4 columns)
        - obj_scale (float)
        - gripper_id
        - gripper_performance
        - gripper_design_loss
        - gripper_loss
    
    Output per sample:
        {
            "grippers": Gripper Encoding + End Prob Mask value - [sample, gripper_dim_mask, finger*segments]
            "object_embedding": Embeddings of objects - [sample,feature_dim, n_ctx]
            "weights": The sampling weight based on the gripper performance - [sample]
        }
        
    Process:
        - Extract all the unique (obj,quat,scale) items from the csv dataset up to the number of max_cond_obj
        - During sampling:
            - Sample randomly from the unique obj set.
            - Among the selected items, averaging the gripper_ids performance (based on loss values) over them.
            - Use the averaged performance for weighted sampling the gripper_id.
    """
    
    def __init__(
        self,csv_path:str,
        object_encoder_name: Literal['placeholder'],
        max_cond_obj: int,
        dl_alpha:float,
        dl_eps:float,
        
        gripper_dir:str,
        gripper_per_sample:int
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.object_encoder = self._init_obj_encoder(object_encoder_name)
        self.max_cond_obj = max_cond_obj
        self.dl_alpha = dl_alpha
        self.dl_eps = dl_eps
        
        self.gripper_dir = gripper_dir
        self.gripper_per_sample = gripper_per_sample
        
        self.object_set,self.gripper_id_set = self._init_set()
        assert self.max_cond_obj <= len(self.object_set)
        
    def _init_set(self):
        df_obj_subset = self.df[["object_id", "quat_w", "quat_x", "quat_y", "quat_z", "scale"]]
        df_unique = df_obj_subset.drop_duplicates(subset=["object_id", "quat_w", "quat_x", "quat_y", "quat_z", "scale"])
        object_configs: List[ObjectConfig] = [
            {
                "name": str(row.object_id),
                "quat": [float(row.quat_w), float(row.quat_x), float(row.quat_y), float(row.quat_z)],
                "scale": float(row.scale)
            }
            for _, row in df_unique.iterrows()
        ]
        return object_configs, set(self.df["index"].unique())
    def _init_obj_encoder(self,object_encoder_name:Literal['placeholder']):
        if object_encoder_name == 'placeholder':
            return EncoderPlaceholder()
        else: raise NotImplementedError(f"This Object Encoder, {object_encoder_name}, has not been implemented")
    
    def __len__(self):
        return sum(math.comb(len(self.object_set), i) for i in range(1, self.max_cond_obj+1))

    def __getitem__(self, index):
        num_obj_samples = np.random.randint(1,self.max_cond_obj+1)
        chosen_obj_idx = np.random.choice(len(self.object_set),(num_obj_samples),replace=False)
        chosen_objs:List[ObjectConfig] = [self.object_set[chosen_idx] for chosen_idx in chosen_obj_idx]
        
        df_subset = pd.DataFrame()
        for obj in chosen_objs:
            mask = (
                (self.df["object_id"].astype(str) == obj["name"]) &
                (self.df["quat_w"] == obj["quat"][0]) &
                (self.df["quat_x"] == obj["quat"][1]) &
                (self.df["quat_y"] == obj["quat"][2]) &
                (self.df["quat_z"] == obj["quat"][3]) &
                (self.df["scale"] == obj["scale"])
            )
            df_subset = pd.concat([df_subset, self.df[mask]], ignore_index=True)
        df_avg = df_subset.groupby("index", as_index=False)[["all_loss"]].mean(numeric_only=True)
        
        weights = 1.0/ (df_avg['all_loss']+self.dl_eps) ** self.dl_alpha
        weights /= weights.sum()

        chosen_gripper_idx = np.random.choice(df_avg['index'],size=self.gripper_per_sample,p=weights)
        
        gripper_data_total = []
        for gripper_idx in chosen_gripper_idx:
            gripper = np.load(os.path.join(self.gripper_dir,f'gripper_nf4_id{gripper_idx}.npz'))
            gripper_data = np.concatenate([
                gripper['gripper_emb'],         # [finger,segment,dim]
                gripper['end_prob'][:,:,None]   # [finger,segment,1]
            ],axis=-1)                          # [finger,segment,dim+1]
            
            F,S,D = gripper_data.shape
            gripper_data_tensor = torch.from_numpy(gripper_data).reshape(-1,D).T
            gripper_data_total.append(gripper_data_tensor)
            
        return {
            "grippers":torch.stack(gripper_data_total),
            "object_embedding": self.object_encoder.encode(chosen_objs,self.gripper_per_sample),
            "weights": torch.from_numpy(np.array(weights[chosen_gripper_idx]))
        }
        

if __name__ == "__main__":
    ds = GripperDataset(
        csv_path='data/grippers/data.csv',
        object_encoder_name = 'placeholder',
        max_cond_obj = 1,
        dl_alpha=20.0,
        dl_eps = 1e-6,
        gripper_dir='data/grippers/',
        gripper_per_sample=10
    )
    dl = DataLoader(ds,batch_size=5,shuffle=True,drop_last=False,num_workers=2)
    for item in dl:
        breakpoint()
        print(item)
    # print(dataloader.gripper_id_set)