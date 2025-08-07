from yacs.config import CfgNode as CN
from typing import Union,Optional,Any
import os

def merge_cfg(base_cfg: CN, cfg: Union[CN, str], replace: Optional[bool] = False):
    assert isinstance(base_cfg, CN)
    assert isinstance(cfg, (CN, str))
    
    cfg_out = base_cfg.clone()

    if isinstance(cfg, str):
        assert os.path.exists(cfg)
        cfg_out.merge_from_file(cfg)
    else:
        cfg_out.merge_from_other_cfg(cfg)

    base_types = (int, float, str, type(None))
    def _dict_to_cfg(_cfg): # recursively convert dict to CN
        if isinstance(_cfg, (list, tuple)):
            for _i, _v in enumerate(_cfg):
                if not isinstance(_v, base_types):
                    _dict_to_cfg(_v)
                if isinstance(_v, dict):
                    _cfg[_i] = CN(_v)
        else:
            for _k, _v in _cfg.items():
                if not isinstance(_v, base_types):
                    _dict_to_cfg(_v)
                if isinstance(_v, dict):
                    _cfg[_k] = CN(_v)
    
    _dict_to_cfg(cfg_out)

    if replace:
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(cfg_out)
        cfg.set_new_allowed(False)
    
    return cfg_out

def set_cfg_attr(cfg: CN, key: str, val: Any):
    fields = key.split('.')
    pointer = cfg
    for x in fields[:-1]:
        pointer = getattr(pointer, x)
    setattr(pointer, fields[-1], val)