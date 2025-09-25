import torch
import numpy as np
import random

import copy
import yaml
import os
from typing import Dict
from tqdm import tqdm
import json

from config.config_dataclass import GeneralConfig

config_path = "config/debug.yaml"
with open(config_path) as f:
    general_config:GeneralConfig = yaml.safe_load(f)

random.seed(general_config['seed'])
np.random.seed(general_config['seed'])
torch.manual_seed(general_config['seed'])

finger_device = torch.device(general_config['sap_config']['device'])
dense_gripper = torch.from_numpy(np.load('asset/finger_0.npy')).to(finger_device) 

from utils import init_log_dir
LOG_PATH_DICT = init_log_dir(
    out_dir = general_config['out_dir'],
    exp_name = general_config['exp_name'],
    tensorboard_log_dir=general_config['tensorboard_log_dir'],
    increment_step=1.
)

def main(num_fingers:int, env_config_file:str):
    from diff_conditioning import CondSet,SoftzooSimulation

    softzoo_config:Dict = general_config['softzoo_config']
    mod_softzoo_config = copy.deepcopy(softzoo_config)
    mod_softzoo_config['num_fingers'] = num_fingers
    mod_softzoo_config['env_config_file'] = env_config_file
    mod_softzoo_config['out_dir'] = os.path.join('./logs',f'num_finger_{num_fingers}',env_config_file[:-5])
    os.makedirs(mod_softzoo_config['out_dir'],exist_ok=True)
    with open(os.path.join(mod_softzoo_config['out_dir'],'softzoo_config.yaml'),'w') as f:
        yaml.safe_dump(mod_softzoo_config,f)
    full_softzoo_config = SoftzooSimulation.load_config(
        cfg_item = mod_softzoo_config
    )
    
    
    general_config['sap_config']['train']['dir_mesh'] = LOG_PATH_DICT['sap_mesh_dir']
    general_config['sap_config']['train']['dir_pcl'] = LOG_PATH_DICT['sap_pcl_dir']
    general_config['sap_config']['train']['dir_train'] = LOG_PATH_DICT['sap_training_dir']

    sim = SoftzooSimulation.init_cond(
        config = general_config['cond_config'][0],
        softzoo_config = full_softzoo_config,
        sap_config=general_config['sap_config']
    )
    ep_reward,reward_log = sim.forward_sim(
        dense_gripper,
        -1,-1,-1
    )
    with open(os.path.join(mod_softzoo_config['out_dir'],'reward.json'),'w') as f:
        json.dump({
            "reward":ep_reward
        },f)
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_finger',type=int,help="Number of fingers")
    parser.add_argument('--env_config_path',type=str,help='Path to environment config .yaml file')
    
    args = parser.parse_args()
    main(
        num_fingers=args.num_finger,
        env_config_file=args.env_config_path
    )