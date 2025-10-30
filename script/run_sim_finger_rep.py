import torch
import numpy as np
import random
import torch.optim as optim

import yaml
import os
from typing import Dict
from tqdm import tqdm

from config.config_dataclass import GeneralConfig

import shutil
config_path = 'config/debug_sim.yaml'
with open(config_path) as f:
    general_config:GeneralConfig = yaml.safe_load(f)

from utils import init_log_dir
LOG_PATH_DICT = init_log_dir(
    out_dir = general_config['out_dir'],
    exp_name = general_config['exp_name'],
    tensorboard_log_dir=general_config['tensorboard_log_dir'],
    increment_step=1.
)
from logger import TENSORBOARD_LOGGER
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
# ctrl_tensor = torch.tensor([1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,1.0,1.0])
# ctrl_tensor = ctrl_tensor.repeat(4,6,1)
raw_tensor = torch.randn(4,10,10)
end_prob_mask = torch.ones(4,10)

# raw_tensor = torch.log(ctrl_tensor/(1-ctrl_tensor))
raw_tensor.requires_grad_(True)
end_prob_mask.requires_grad_(True)
lr = 1e-1
optim = optim.Adam([raw_tensor,end_prob_mask], lr=lr)

for i in tqdm(range(100)):
    TENSORBOARD_LOGGER.log_scalar('Simulation/Encoding_Norm',raw_tensor.flatten().norm(2))
    sigmoid_tensor = torch.sigmoid(raw_tensor)
    optim.zero_grad()
    sim_cls.calculate_gradient(
        sigmoid_tensor,
        end_prob_mask,
        i
    )
    optim.step()
    TENSORBOARD_LOGGER.increment()