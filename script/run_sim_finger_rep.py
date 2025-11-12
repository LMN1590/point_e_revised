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

from diff_conditioning.simulation_env.softzoo_final import SoftZooSimulation

softzoo_config:Dict = general_config['softzoo_config']
full_softzoo_config = SoftZooSimulation.load_config(
    cfg_item = softzoo_config
)
full_softzoo_config.out_dir = LOG_PATH_DICT['softzoo_log_dir']

sim_cls=SoftZooSimulation(
    config = full_softzoo_config,
    num_fingers=4,
    max_num_segments=10,
    gripper_dim=11,
    **general_config['cond_config'][0]
)


ctrl_tensor = torch.tensor([1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,1.0,1.0,0.0])
ctrl_tensor = ctrl_tensor.repeat(4,10,1)
# raw_tensor = torch.randn(4,10,10)
end_prob_mask = torch.ones(4,10)

# raw_tensor = torch.log(ctrl_tensor/(1-ctrl_tensor))
# raw_tensor.requires_grad_(True)
# end_prob_mask.requires_grad_(True)
# lr = 1e-1
# optim = optim.Adam([raw_tensor,end_prob_mask], lr=lr)

for i in tqdm(range(1)):
    # TENSORBOARD_LOGGER.log_scalar('Simulation/Encoding_Norm',raw_tensor.flatten().norm(2))
    # optim.zero_grad()
    sim_cls.forward_sim(
        ctrl_tensor,
        end_prob_mask,
        i,0,0
    )
    # optim.step()
    # TENSORBOARD_LOGGER.increment()