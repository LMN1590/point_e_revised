import torch

from tqdm.auto import tqdm
import yaml
from typing import Dict
import os

from custom_diffusion.diffusion.configs import diffusion_from_config
from custom_diffusion.models.configs import model_from_config
from custom_diffusion.config import DIFFUSION_CONFIGS,MODEL_CONFIGS

from custom_diffusion.diffusion.gripper_sampler import GripperSampler

from diff_conditioning.simulation_env.softzoo_final import SoftZooSimulation
from config.config_dataclass import GeneralConfig

if __name__ == "__main__":
    config_path = 'config/benchmark_sim_topdown.yaml'
    with open(config_path) as f:
        general_config:GeneralConfig = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gripper_model = 'gripper_rep'
    gripper_diffusion = 'custom_finger_diffusion'
    
    base_model = model_from_config(MODEL_CONFIGS[gripper_model],device).eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[gripper_diffusion])
    base_model._init_fingers_topo(
        gripper_dim = 10,
        max_num_segments = 10,
        num_fingers = 4
    )
    
    sampler = GripperSampler(
        device = device,
        models = [base_model],
        diffusions = [base_diffusion],
        model_kwargs_key_filter=['objects'],
        guidance_scale = [3.0],
        clip_denoised=True,
        sampling_mode = 'ddpm',
        
        gripper_dim= 10,
        max_num_segments=10,
        num_fingers=4
    )
    pre_noise = torch.randn((2, 11, 40)).to(device)
    
    softzoo_config:Dict = general_config['softzoo_config']
    full_softzoo_config = SoftZooSimulation.load_config(
        cfg_item = softzoo_config
    )
    full_softzoo_config.out_dir = 'logs/placeholder_logs'
    os.makedirs(full_softzoo_config.out_dir,exist_ok=True)
    sim_cls = SoftZooSimulation(
        full_softzoo_config,
        name = general_config['cond_config'][0]['name'],
        grad_scale=general_config['cond_config'][0]['grad_scale'],
        calc_gradient=general_config['cond_config'][0]['calc_gradient'],
        grad_clamp=general_config['cond_config'][0]['grad_clamp'],
        logging_bool=general_config['cond_config'][0]['logging_bool'],
    )
    
    
    for x in tqdm(sampler.sample_batch_progressive(
        batch_size=1,
        model_kwargs=dict(
            # embeddings = torch.zeros(1,128,100).to(device),
            objects= ['test']
        ),
        pre_noise = [pre_noise],
        cond_fn_lst= [sim_cls.calculate_gradient]
    )):
        pass
    

    # output = base_model(
    #     test_input,
    #     t=torch.tensor([10.]).to(device),
    #     embeddings = 
    # )
    # print(output.shape)
    