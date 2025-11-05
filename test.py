import torch
from tqdm.auto import tqdm

from custom_diffusion.diffusion.configs import diffusion_from_config
from custom_diffusion.models.configs import model_from_config
from custom_diffusion.config import DIFFUSION_CONFIGS,MODEL_CONFIGS

from custom_diffusion.diffusion.gripper_sampler import GripperSampler

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gripper_model = 'gripper_rep'
    gripper_diffusion = 'custom_finger_diffusion'
    
    base_model = model_from_config(MODEL_CONFIGS[gripper_model],device)
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
    pre_noise = [torch.randn((2, 11, 40)).to(device)]
    
    for x in tqdm(sampler.sample_batch_progressive(
        batch_size=1,
        model_kwargs=dict(
            # embeddings = torch.zeros(1,128,100).to(device),
            objects= ['test']
        ),
        pre_noise = pre_noise,
        cond_fn_lst= [None]
    )):
        pass
    

    # output = base_model(
    #     test_input,
    #     t=torch.tensor([10.]).to(device),
    #     embeddings = 
    # )
    # print(output.shape)
    