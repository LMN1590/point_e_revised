from diff_conditioning.simulation_env import SoftzooSimulation

import torch
import numpy as np
import random

import yaml
import os
from typing import Dict
from tqdm import tqdm

from config.config_dataclass import GeneralConfig

config_path = "config/debug.yaml"

with open(config_path) as f:
    general_config:GeneralConfig = yaml.safe_load(f)
    
random.seed(general_config['seed'])
np.random.seed(general_config['seed'])
torch.manual_seed(general_config['seed'])

finger_device = torch.device(general_config['sap_config']['device'])
dense_gripper = torch.from_numpy(np.load('finger_0.npy')).to(finger_device)   

from utils import init_log_dir
LOG_PATH_DICT = init_log_dir(
    out_dir = general_config['out_dir'],
    exp_name = general_config['exp_name'],
    tensorboard_log_dir=general_config['tensorboard_log_dir'],
    increment_step=1.
)

from diff_conditioning import CondSet,SoftzooSimulation

softzoo_config:Dict = general_config['softzoo_config']
full_softzoo_config = SoftzooSimulation.load_config(
    cfg_item = softzoo_config
)
full_softzoo_config.out_dir = LOG_PATH_DICT['softzoo_log_dir']
general_config['sap_config']['train']['dir_mesh'] = LOG_PATH_DICT['sap_mesh_dir']
general_config['sap_config']['train']['dir_pcl'] = LOG_PATH_DICT['sap_pcl_dir']
general_config['sap_config']['train']['dir_train'] = LOG_PATH_DICT['sap_training_dir']

sim = SoftzooSimulation.init_cond(
    config = general_config['cond_config'][0],
    softzoo_config = full_softzoo_config,
    sap_config=general_config['sap_config']
)
ep_reward = sim.forward_sim(
    dense_gripper,
    -1,-1,-1
)
all_loss,grad,grad_name_control = sim.backward_sim()