from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    DeviceStatsMonitor,
    EarlyStopping
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

from typing import List,Literal,Optional,Dict
from typing_extensions import TypedDict
import os
from datetime import datetime

from .trainer import DiffusionTrainer
from .dataloader import GripperDataset

from custom_diffusion.models.transformer.gripper_rep_diffusion_transformer import GripperRepDiffusionTransformer
from custom_diffusion.models.object_encoder import EncoderPlaceholder
from custom_diffusion.diffusion.gaussian_diffusion.gaussian_diff_class import GaussianDiffusion

from custom_diffusion.diffusion.configs import diffusion_from_config
from custom_diffusion.models.configs import model_from_config
from custom_diffusion.config import DIFFUSION_CONFIGS,MODEL_CONFIGS

CALLBACKS = {
    "LearningRateMonitor": LearningRateMonitor,
    "ModelCheckpoint": ModelCheckpoint,
    "RichProgressBar": RichProgressBar,
    "DeviceStatsMonitor": DeviceStatsMonitor,
    "EarlyStopping": EarlyStopping
}

class GripperConfig(TypedDict):
    gripper_dim:int
    max_num_segments:int
    num_fingers:int

class DiffusionTrainerConfig(TypedDict):
    ema_power:float
    ema_update_after_step:int
    
    learning_rate: float
    lr_warmup_steps:int

    acc_threshold: float

class DatasetConfig(TypedDict):
    csv_path:str
    object_encoder_name:Literal['placeholder']
    max_cond_obj:int
    gripper_dir: str
    gripper_per_sample:int
    
class TrainConfig(TypedDict):
    exp_name:str
    
    diffusion_config:DiffusionTrainerConfig
    dataset_config:DatasetConfig
    gripper_config:GripperConfig
    
    callbacks_config:Dict[str,Dict]
    
    logger_type: List[Literal['tensorboard','wandb','csv']]
    gpus: List[int]
    check_val_every_n_epoch:int
    log_every_n_steps:int
    max_epochs:int
    default_root_dir:str
    
    dl_batch_size:int
    dl_num_workers:int
    
    noise_pred_net:str
    diffusion_type:str

def train(config:TrainConfig,existing_ckpt_path:Optional[str] = None):
    default_log_dir = os.path.join(config['default_root_dir'],f"{config['exp_name']}_{datetime.now()}")
    ds = GripperDataset(**config['dataset_config'])
    dl = DataLoader(
        ds,
        batch_size = config['dl_batch_size'],
        num_workers= config['dl_num_workers'],
        shuffle=True, drop_last=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = model_from_config(MODEL_CONFIGS[config['noise_pred_net']],device).eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[config['diffusion_type']])
    base_model._init_fingers_topo(**config['gripper_config'])
    diff_trainer = DiffusionTrainer(
        noise_pred_net=base_model,
        diffusion=base_diffusion,
        num_epochs = config['max_epochs'],
        **config['diffusion_config']
    )
    
    callbacks = [
        CALLBACKS[k](
            **(v|{
                "dirpath": os.path.join(default_log_dir,'checkpoints')
            } if k == 'ModelCheckpoint' else {})
        ) for k,v in config['callbacks_config'].items() if k in CALLBACKS.keys()
    ]
    
    
    loggers = []
    for log_type in config['logger_type']:
        if log_type == 'tensorboard':
            loggers.append(TensorBoardLogger(
                save_dir=default_log_dir,
                name = f"tensorboard_log_{config['exp_name']}",
            ))
        elif log_type == 'wandb':
            loggers.append(WandbLogger(
                name = config['exp_name'],
                project = f"wandb_log_{config['exp_name']}",
                save_dir = default_log_dir
            ))
        elif log_type == 'csv':
            loggers.append(CSVLogger(
                save_dir=default_log_dir,
                name = f"csv_log_{config['exp_name']}",
            ))
            
    
    trainer = LightningTrainer(
        accelerator='gpu',
        devices = config['gpus'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        log_every_n_steps=config['log_every_n_steps'],
        max_epochs=config['max_epochs'],
        default_root_dir=default_log_dir,
        callbacks=callbacks,
        logger=loggers
    )
    
    if existing_ckpt_path is not None:
        print('loading diffusion checkpoint from', existing_ckpt_path)
        trainer.fit(diff_trainer,dl,dl,ckpt_path=existing_ckpt_path)
    else:
        trainer.fit(diff_trainer,dl,dl)

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path",type=str,help='Path to the training config .yaml file')
    parser.add_argument("--ckpt_path",type=str,help = 'Continuous training of existing ckpt file',default=None)

    args = parser.parse_args()
    config_file = os.path.join('config','training_config',args.train_config_path)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    train(config,args.ckpt_path)