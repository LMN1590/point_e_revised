import torch
import numpy as np
import random

import yaml
import os
from typing import Dict

from config.config_dataclass import GeneralConfig

import shutil
config_path = 'config/config.yaml'
with open(config_path) as f:
    general_config:GeneralConfig = yaml.safe_load(f)

from utils import init_log_dir
LOG_PATH_DICT = init_log_dir(
    out_dir = general_config['out_dir'],
    exp_name = general_config['exp_name'],
    tensorboard_log_dir=general_config['tensorboard_log_dir'],
    increment_step=1.
)
shutil.copyfile(config_path, os.path.join(LOG_PATH_DICT['exp_dir'],'config.yaml'))

random.seed(general_config['seed'])
np.random.seed(general_config['seed'])
torch.manual_seed(general_config['seed'])

from diff_conditioning.simulation_env.alt_softzoo_sim import AltSoftzooSimulation

softzoo_config:Dict = general_config['softzoo_config']
full_softzoo_config = AltSoftzooSimulation.load_config(
    cfg_item = softzoo_config
)
full_softzoo_config.out_dir = LOG_PATH_DICT['softzoo_log_dir']
general_config['sap_config']['train']['dir_mesh'] = LOG_PATH_DICT['sap_mesh_dir']
general_config['sap_config']['train']['dir_pcl'] = LOG_PATH_DICT['sap_pcl_dir']
general_config['sap_config']['train']['dir_train'] = LOG_PATH_DICT['sap_training_dir']

sim_cls = AltSoftzooSimulation.init_cond(
    config = general_config['cond_config'][0],
    softzoo_config=full_softzoo_config,
    sap_config=general_config['sap_config']
)

ctrl_tensor = torch.sigmoid(torch.randn(4,4,10))
sim_cls.calculate_gradient(
    ctrl_tensor
)