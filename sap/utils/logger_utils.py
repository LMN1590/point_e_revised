import os

from config_dataclass import SAPConfig

def initialize_logger(cfg:SAPConfig):
    out_dir = cfg['train']['out_dir']
    if not out_dir:
        os.makedirs(out_dir)

    if cfg['train']['exp_mesh']:
        cfg['train']['dir_mesh'] = os.path.join(out_dir, 'vis/mesh')
        os.makedirs(cfg['train']['dir_mesh'], exist_ok=True)
    if cfg['train']['exp_pcl']:
        cfg['train']['dir_pcl'] = os.path.join(out_dir, 'vis/pointcloud')
        os.makedirs(cfg['train']['dir_pcl'], exist_ok=True)