from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer as LightningTrainer

from .trainer import DiffusionTrainer
from .dataloader import GripperDataset

from custom_diffusion.models.transformer.gripper_rep_diffusion_transformer import GripperRepDiffusionTransformer
from custom_diffusion.models.object_encoder import EncoderPlaceholder
from custom_diffusion.diffusion.gaussian_diffusion.gaussian_diff_class import GaussianDiffusion

from custom_diffusion.diffusion.configs import diffusion_from_config
from custom_diffusion.models.configs import model_from_config
from custom_diffusion.config import DIFFUSION_CONFIGS,MODEL_CONFIGS

ds = GripperDataset(
    csv_path='data/grippers/data.csv',
    object_encoder=EncoderPlaceholder(),
    max_cond_obj = 1,
    gripper_dir='data/grippers/',
    gripper_per_sample=10
)
dl = DataLoader(
    ds,batch_size = 10,
    shuffle=True,
    num_workers=4,
    drop_last=False
)

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
diff_trainer = DiffusionTrainer(
    noise_pred_net=base_model,
    diffusion_scheduler=base_diffusion,
)
trainer = LightningTrainer(
    accelerator='gpu',
    devices = [1],
)
trainer.fit(
    diff_trainer,dl,dl
)