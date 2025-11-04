import torch
from tqdm.auto import tqdm

from custom_diffusion.diffusion.configs import diffusion_from_config
from custom_diffusion.models.configs import model_from_config
from custom_diffusion.config import DIFFUSION_CONFIGS,MODEL_CONFIGS

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gripper_model = 'gripper_rep'
    gripper_diffusion = 'custom_finger_diffusion'
    
    base_model = model_from_config(MODEL_CONFIGS[gripper_model],device)
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[gripper_diffusion])
    