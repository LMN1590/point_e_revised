# python -m benchmark.benchmark --num_finger 2 --env_config_path benchmark_lifting.yaml

import torch
import numpy as np
import random

import copy
import yaml
import os
from typing import Dict
from tqdm import tqdm
import json

from diff_conditioning.simulation_env.alt_softzoo_sim import AltSoftzooSimulation
from config.config_dataclass import GeneralConfig

config_path = "config/benchmark_sim_topdown.yaml"
with open(config_path) as f:
    general_config:GeneralConfig = yaml.safe_load(f)

random.seed(general_config['seed'])
np.random.seed(general_config['seed'])
torch.manual_seed(general_config['seed'])

from utils import init_log_dir
LOG_PATH_DICT = init_log_dir(
    out_dir = general_config['out_dir'],
    exp_name = general_config['exp_name'],
    tensorboard_log_dir=general_config['tensorboard_log_dir'],
    increment_step=1.
)

def generate_simcls(env_config_file:str):
    softzoo_config:Dict = general_config['softzoo_config']
    mod_softzoo_config = copy.deepcopy(softzoo_config)
    mod_softzoo_config['env_config_file'] = env_config_file
    mod_softzoo_config['out_dir'] = os.path.join('./logs',env_config_file[:-5])
    os.makedirs(mod_softzoo_config['out_dir'],exist_ok=True)
    with open(os.path.join(mod_softzoo_config['out_dir'],'softzoo_config.yaml'),'w') as f:
        yaml.safe_dump(mod_softzoo_config,f)
    full_softzoo_config = AltSoftzooSimulation.load_config(
        cfg_item = mod_softzoo_config
    )
    
    general_config['sap_config']['train']['dir_mesh'] = LOG_PATH_DICT['sap_mesh_dir']
    general_config['sap_config']['train']['dir_pcl'] = LOG_PATH_DICT['sap_pcl_dir']
    general_config['sap_config']['train']['dir_train'] = LOG_PATH_DICT['sap_training_dir']

    sim_cls = AltSoftzooSimulation.init_cond(
        config = general_config['cond_config'][0],
        softzoo_config = full_softzoo_config,
        sap_config=general_config['sap_config']
    )
    
    return sim_cls

def run_benchmark_gripping(
    sim_cls:AltSoftzooSimulation,
    gripper_emb:torch.Tensor,
    end_prob_mask:torch.Tensor,
    index:int
):
    ep_reward,reward_log,design_loss = sim_cls.forward_sim(
        ctrl_tensor = gripper_emb,
        end_prob_mask = end_prob_mask,
        batch_idx=index,
        sampling_step = -1,
        local_iter = -1
    )
    all_loss,grad,grad_name_control = sim_cls.backward_sim()
    with open(os.path.join(sim_cls.config.out_dir,f'gripping_result_id{index}.json'),'w') as f:
        json.dump({
            "reward":ep_reward,
            "design_loss":design_loss,
            "all_loss":all_loss
        },f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gripper_dir',type=str,help='Directory containing gripper embeddings')
    parser.add_argument('--env_config_path',type=str,help='Path to environment config .yaml file')
    
    args = parser.parse_args()
    sim_cls = generate_simcls(env_config_file=args.env_config_path)
    num_grippers = len(os.listdir(args.gripper_dir))
    
    for i in tqdm(range(num_grippers)):
        gripper_path = os.path.join(args.gripper_dir,f'gripper_nf4_id{i}.npz')
        gripper_item = np.load(gripper_path)
        
        run_benchmark_gripping(
            sim_cls = sim_cls,
            gripper_emb = torch.from_numpy(gripper_item['gripper_emb']),
            end_prob_mask = torch.from_numpy(gripper_item['end_prob']),
            index = i,
        )