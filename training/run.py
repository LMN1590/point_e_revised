from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning import seed_everything
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
import shutil
from datetime import date

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
    lr_warmup_percentage:int
    warmup_min_lr_ratio: float
    min_lr_ratio: float

    acc_threshold: float

class DatasetConfig(TypedDict):
    csv_path:str
    object_encoder_name:Literal['placeholder']
    max_cond_obj:int
    dl_alpha:float
    dl_eps:float
    gripper_dir: str
    gripper_per_sample:int
    sample_mode:Literal['random','top']
    
class TrainConfig(TypedDict):
    exp_name:str
    seed:int
    
    train_dataset_config:DatasetConfig
    val_dataset_config:DatasetConfig
    
    diffusion_config:DiffusionTrainerConfig
    gripper_config:GripperConfig
    
    callbacks_config:Dict[str,Dict]
    
    logger_type: List[Literal['tensorboard','wandb','csv']]
    gpus: List[int]
    check_val_every_n_epoch:int
    log_every_n_steps:int
    max_epochs:int
    default_root_dir:str
    gradient_clip_val: float
    gradient_clip_algorithm: str
    num_sanity_val_steps: int
    
    dl_total_batch_size:int
    dl_micro_batch_size:int
    dl_num_workers:int
    
    noise_pred_net:str
    diffusion_type:str

def train(config:TrainConfig,existing_ckpt_path:Optional[str] = None,id:Optional[str] = None):
    seed_everything(config['seed'],workers=True)
    
    default_log_dir = os.path.join(
        config['default_root_dir'],
        config['exp_name'],
        str(id)
    )
    os.makedirs(default_log_dir,exist_ok=True)
    with open(os.path.join(default_log_dir,'config.yaml'),'w') as f:
        yaml.safe_dump(config,f)
    train_ds = GripperDataset(**config['train_dataset_config'])
    train_dl = DataLoader(
        train_ds,
        batch_size = config['dl_micro_batch_size'],
        num_workers= config['dl_num_workers'],
        shuffle=True, drop_last=False
    )
    val_ds = GripperDataset(**config['val_dataset_config'])
    val_dl = DataLoader(
        val_ds,
        batch_size = config['dl_micro_batch_size'],
        num_workers= config['dl_num_workers'],
        shuffle=True, drop_last=False
    )

    base_model = model_from_config(MODEL_CONFIGS[config['noise_pred_net']],torch.device('cpu'))
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[config['diffusion_type']])
    base_model._init_fingers_topo(**config['gripper_config'])
    
    with open(os.path.join(default_log_dir,'model_config.yaml'),'w') as f:
        yaml.safe_dump(MODEL_CONFIGS[config['noise_pred_net']],f)
    with open(os.path.join(default_log_dir,'diffusion_config.yaml'),'w') as f:
        yaml.safe_dump(DIFFUSION_CONFIGS[config['diffusion_type']],f)
    
    pcd_log_dir = os.path.join(default_log_dir,'pcd')
    os.makedirs(pcd_log_dir,exist_ok=True)
    
    diff_trainer = DiffusionTrainer(
        noise_pred_net=base_model,
        diffusion=base_diffusion,
        num_epochs = config['max_epochs'],
        **config['diffusion_config'],
        **config['gripper_config'],
        pcd_log_dir=pcd_log_dir,
        total_num_steps=int(config['max_epochs']*len(train_ds)/config['dl_total_batch_size'])
    )
    
    callbacks = [
        CALLBACKS[k](
            **(v|({
                "dirpath": os.path.join(default_log_dir,'checkpoints')
            } if k == 'ModelCheckpoint' else {}))
        ) for k,v in config['callbacks_config'].items() if k in CALLBACKS.keys()
    ]
    
    
    loggers = []
    for log_type in config['logger_type']:
        if log_type == 'tensorboard':
            loggers.append(TensorBoardLogger(
                save_dir=config['default_root_dir'],
                name = config['exp_name'],
                version = id,
            ))
        elif log_type == 'wandb':
            loggers.append(WandbLogger(
                save_dir=config['default_root_dir'],
                
                name = config['exp_name'],
                version = id,
                project = "Gripper Diffusion",                
            ))
        elif log_type == 'csv':
            loggers.append(CSVLogger(
                save_dir=config['default_root_dir'],
                name = config['exp_name'],
                version = id,
            ))
            
    
    trainer = LightningTrainer(
        accelerator='gpu',
        devices = config['gpus'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        log_every_n_steps=config['log_every_n_steps'],
        max_epochs=config['max_epochs'],
        default_root_dir=default_log_dir,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val = config['gradient_clip_val'],
        gradient_clip_algorithm = config['gradient_clip_algorithm'],
        num_sanity_val_steps = config['num_sanity_val_steps'],
        accumulate_grad_batches=config['dl_total_batch_size'] // (config['dl_micro_batch_size'] * len(config['gpus'])),
    )
    
    if existing_ckpt_path is not None:
        print('loading diffusion checkpoint from', existing_ckpt_path)
        trainer.fit(diff_trainer,train_dl,val_dl,ckpt_path=existing_ckpt_path)
    else:
        trainer.fit(diff_trainer,train_dl,val_dl)

if __name__ == "__main__":
    import argparse
    import yaml
    from uuid import uuid4 as uuid
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path",type=str,help='Path to the training config .yaml file')
    parser.add_argument("--ckpt_path",type=str,help = 'Continuous training of existing ckpt file',default=None)
    parser.add_argument("--run_id",type=str,help = "Run ID to continue training", default=None)

    args = parser.parse_args()
    
    config_file = os.path.join('training','training_config',args.train_config_path)
    with open(config_file) as f:
        config:TrainConfig = yaml.safe_load(f)
    
    run_id = f"{str(date.today()).replace('-','')}_{'' if args.run_id is None else args.run_id}_{str(uuid())[-6:]}_seed{config['seed']}"
    
    train(config,args.ckpt_path,run_id)