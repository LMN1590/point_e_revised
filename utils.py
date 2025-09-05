import torch

import os
from typing import TypedDict,Tuple,Callable

from logger import init_all_logger

class LogPathDict(TypedDict):
    exp_dir:str
    
    pointe_npz_dir:str
    pointe_ply_dir:str
    
    softzoo_log_dir:str
    
    sap_training_dir:str
    sap_mesh_dir:str
    sap_pcl_dir:str
    
    embedding_dir:str

def init_log_dir(out_dir:str, exp_name:str, tensorboard_log_dir:str,increment_step:float)->LogPathDict:
    exp_dir = os.path.join(out_dir,exp_name)
    os.makedirs(exp_dir,exist_ok=True)
    
    pointe_log_dir = os.path.join(exp_dir,'pointe')
    pointe_npz_dir = os.path.join(pointe_log_dir,'npz')
    pointe_ply_dir = os.path.join(pointe_log_dir,'ply')
    os.makedirs(pointe_npz_dir,exist_ok=True)
    os.makedirs(pointe_ply_dir,exist_ok=True)
    
    softzoo_log_dir = os.path.join(exp_dir,'softzoo')
    os.makedirs(softzoo_log_dir,exist_ok=True)
    
    sap_log_dir = os.path.join(exp_dir,'sap')
    sap_training_dir = os.path.join(sap_log_dir,'training')
    sap_mesh_dir = os.path.join(sap_log_dir,'mesh')
    sap_pcl_dir = os.path.join(sap_log_dir,'pcl')
    os.makedirs(sap_mesh_dir,exist_ok=True)
    os.makedirs(sap_pcl_dir,exist_ok=True)
    os.makedirs(sap_training_dir,exist_ok=True)
    
    embedding_dir = os.path.join(exp_dir,'embeddings')
    os.makedirs(embedding_dir,exist_ok=True)
    
    init_all_logger(out_dir, exp_name,tensorboard_log_dir,increment_step=increment_step)
    
    return {
        "exp_dir":exp_dir,
        "pointe_npz_dir":pointe_npz_dir,
        "pointe_ply_dir":pointe_ply_dir,
        
        "softzoo_log_dir":softzoo_log_dir,

        'sap_training_dir': sap_training_dir,
        "sap_mesh_dir":sap_mesh_dir,
        "sap_pcl_dir":sap_pcl_dir,
        "embedding_dir":embedding_dir
    }
    
CLIP_SIZE = (224,224)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None,:,None,None]
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None,:,None,None]
def sample_random_CLIP_emb(batch_size:int,add_emb_func:Callable[[torch.Tensor],torch.Tensor]):
    init_sample = torch.rand(batch_size,3,*CLIP_SIZE)
    return add_emb_func((init_sample - CLIP_MEAN)/CLIP_STD)
    # return init_sample