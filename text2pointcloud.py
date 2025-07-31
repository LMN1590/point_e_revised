import torch

from tqdm.auto import tqdm

from point_e.diffusion.configs import diffusion_from_config
from point_e.models.configs import model_from_config
from point_e.config import MODEL_CONFIGS,DIFFUSION_CONFIGS

from point_e.utils.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.utils.plotting import plot_point_cloud

def load_model(base_model_name:str, upsampler_name:str, device:torch.device):
    base_model = model_from_config(MODEL_CONFIGS[base_model_name], device=device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_model_name])
    base_ckpt = load_checkpoint(
        base_model_name, device=device
    )
    base_model.load_state_dict(base_ckpt)
    
    upsampler_model = model_from_config(MODEL_CONFIGS[upsampler_name], device=device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])
    upsampler_ckpt = load_checkpoint(
        upsampler_name, device=device
    )
    upsampler_model.load_state_dict(upsampler_ckpt)
    
    return [base_model, upsampler_model], [base_diffusion, upsampler_diffusion]
    
    
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, diffusions = load_model("base40M-textvec", "upsample", device)
    sampler = PointCloudSampler(
        device = device,
        models=models,
        diffusions=diffusions,
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )
    
    # Set a prompt to condition on.
    prompt = 'a red motorcycle'

    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
        
    pc = sampler.output_to_point_clouds(samples)[0]
    fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))