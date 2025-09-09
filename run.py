import torch
import numpy as np
import random

import yaml
import os
from typing import Dict
from tqdm import tqdm
import shutil

from config.config_dataclass import GeneralConfig

# region Prepare Configurations and Logger
CONFIG_PATH = 'config/config.yaml'
with open('config/config.yaml') as f:
    general_config:GeneralConfig = yaml.safe_load(f)

from utils import init_log_dir
LOG_PATH_DICT = init_log_dir(
    out_dir = general_config['out_dir'],
    exp_name = general_config['exp_name'],
    tensorboard_log_dir=general_config['tensorboard_log_dir'],
    increment_step=1.
)
shutil.copyfile(CONFIG_PATH, os.path.join(LOG_PATH_DICT['exp_dir'],'config.yaml'))

random.seed(general_config['seed'])
np.random.seed(general_config['seed'])
torch.manual_seed(general_config['seed'])

from logger import TENSORBOARD_LOGGER,CSVLOGGER
# endregion
# region Initialize Point-E models
from point_e.diffusion.configs import diffusion_from_config
from point_e.models.configs import model_from_config
from point_e.config import MODEL_CONFIGS,DIFFUSION_CONFIGS

from point_e.utils.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler


pointe_configs = general_config['pointe_config']
device = torch.device(pointe_configs['device'])

base_name = pointe_configs['base_model_name']
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

base_model.load_state_dict(load_checkpoint(base_name, device))
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

guidance_scale = pointe_configs['guidance_scale']
num_points=pointe_configs['num_points']

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=num_points,
    aux_channels=['R', 'G', 'B'],
    guidance_scale=guidance_scale,
    model_kwargs_key_filter=('embeddings', ''), # Do not condition the upsampler at all #TODO: change this into random sampled embeddings later
    use_karras = (False,False)
)
# endregion

# region Intialize SoftZoo Simulations
from diff_conditioning import CondSet,SoftzooSimulation

softzoo_config:Dict = general_config['softzoo_config']
full_softzoo_config = SoftzooSimulation.load_config(
    cfg_item = softzoo_config
)
full_softzoo_config.out_dir = LOG_PATH_DICT['softzoo_log_dir']
general_config['sap_config']['train']['dir_mesh'] = LOG_PATH_DICT['sap_mesh_dir']
general_config['sap_config']['train']['dir_pcl'] = LOG_PATH_DICT['sap_pcl_dir']
general_config['sap_config']['train']['dir_train'] = LOG_PATH_DICT['sap_training_dir']

if general_config['cond_config'] is None or len(general_config['cond_config']) == 0:
    cond_fn_lst = [None,None]
else:
    cond_set = CondSet(
        cond_config_lst = general_config['cond_config'],
        cond_overall_logging= general_config['cond_overall_logging'],
        softzoo_config = full_softzoo_config,
        sap_config = general_config['sap_config']
    )
    cond_fn_lst = [cond_set.calculate_gradient,None]
    
# endregion

# region Load Noise
from PIL import Image
from utils import sample_random_CLIP_emb,sample_from_img
import shutil
from torchvision import transforms

preload_config = general_config['preload_emb']
if preload_config['condition_embedding']['path'] is not None:
    path = preload_config['condition_embedding']['path']
    if path.split('.')[-1] == 'npy':
        # Read .npy
        conditional_emb_npy = np.load(preload_config['condition_embedding']['path'])
        conditional_emb_tensor = torch.from_numpy(conditional_emb_npy).to(device = device,dtype=torch.float32)
    else:
        # Read Images
        shutil.copy(path,os.path.join(LOG_PATH_DICT['embedding_dir'],f'conditional_img.{path.split(".")[-1]}'))
        img = Image.open(path)
        conditional_emb_tensor = sample_from_img([img],base_model.clip.embed_images_grid).to(device = device,dtype=torch.float32)
else:
    if preload_config['condition_embedding']['random']:
        init_sample = torch.rand(*preload_config['condition_embedding']['shape'])
    else:
        init_sample = torch.zeros(*preload_config['condition_embedding']['shape'])
    image = transforms.ToPILImage()(init_sample[0]).save(
        os.path.join(LOG_PATH_DICT['embedding_dir'],'conditional_img.png')
    )
    conditional_emb_tensor = sample_random_CLIP_emb(init_sample,base_model.clip.embed_preprocessed_images_grid).to(device = device,dtype=torch.float32)
np.save(os.path.join(LOG_PATH_DICT['embedding_dir'],f'conditional_emb_{"random" if preload_config["condition_embedding"]["random"] else "zero"}.npy'),conditional_emb_tensor.detach().cpu().numpy())

base_noise_npy = np.load(preload_config['diffusion_noise']['path'])  \
    if preload_config['diffusion_noise']['path'] is not None \
    else np.random.randn(*preload_config['diffusion_noise']['shape'])
base_noise_tensor = torch.from_numpy(base_noise_npy).to(device = device,dtype=torch.float32)
np.save(os.path.join(LOG_PATH_DICT['embedding_dir'],'base_noise.npy'),base_noise_npy)

upsample_noise_npy = np.load(preload_config['upsample_noise']['path'])  \
    if preload_config['upsample_noise']['path'] is not None \
    else np.random.randn(*preload_config['upsample_noise']['shape'])
upsample_noise_tensor = torch.from_numpy(upsample_noise_npy).to(device = device,dtype=torch.float32)
np.save(os.path.join(LOG_PATH_DICT['embedding_dir'],'upsample_noise.npy'),upsample_noise_npy)

pre_noise = [base_noise_tensor,upsample_noise_tensor]
# endregion

# region Run Sampling
final_sample:torch.Tensor = None
count = 0
for x in tqdm(sampler.sample_batch_progressive(
    batch_size=1, model_kwargs=dict(embeddings=conditional_emb_tensor),
    pre_noise=pre_noise,
    cond_fn_lst=cond_fn_lst
),position=0):
    if (general_config['total_steps']-count)%general_config['save_every_iter'] == 0: 
        cur_pc = sampler.output_to_point_clouds(x)[0]
        ply_name = f"result_{general_config['total_steps']-count}_{general_config['exp_name']}.ply"
        ply_dir = os.path.join(LOG_PATH_DICT['pointe_ply_dir'],ply_name)
        with open(ply_dir,'wb') as f:
            cur_pc.write_ply(f)

        npz_name = f"result_{general_config['total_steps']-count}_{general_config['exp_name']}.npz"
        npz_dir = os.path.join(LOG_PATH_DICT['pointe_npz_dir'],npz_name)
        cur_pc.save(npz_dir)

    final_sample = x
    count+=1
# endregion

# region Clean Up
TENSORBOARD_LOGGER.close()
CSVLOGGER.close()

from point_e.utils.plotting import plot_point_cloud
pc = sampler.output_to_point_clouds(final_sample)[0]
fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
fig_dir = os.path.join(LOG_PATH_DICT['exp_dir'],'final_result.png')
fig.savefig(fig_dir, dpi=300, bbox_inches="tight")
# endregion