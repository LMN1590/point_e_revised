import torch

import os
from functools import lru_cache
from typing import Optional, Dict
from tqdm.auto import tqdm

import requests
from filelock import FileLock

from .path import MODEL_CACHE_DIR

MODEL_PATHS = {
    "base40M-imagevec": "https://openaipublic.azureedge.net/main/point-e/base_40m_imagevec.pt",
    "base40M-textvec": "https://openaipublic.azureedge.net/main/point-e/base_40m_textvec.pt",
    "base40M-uncond": "https://openaipublic.azureedge.net/main/point-e/base_40m_uncond.pt",
    "base40M": "https://openaipublic.azureedge.net/main/point-e/base_40m.pt",
    "base300M": "https://openaipublic.azureedge.net/main/point-e/base_300m.pt",
    "base1B": "https://openaipublic.azureedge.net/main/point-e/base_1b.pt",
    "upsample": "https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt",
    "sdf": "https://openaipublic.azureedge.net/main/point-e/sdf.pt",
    "pointnet": "https://openaipublic.azureedge.net/main/point-e/pointnet.pt",
}

@lru_cache()
def default_cache_dir() -> str:
    return MODEL_CACHE_DIR

def fetch_file_cached(
    url:str, progress:bool = True,
    cache_dir:Optional[str] = None, chunk_size:int = 4096
):
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(local_path): return local_path
    
    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    
    with FileLock(local_path + ".lock"):
        if progress: pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress: pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress: pbar.close()
        return local_path
    
def load_checkpoint(
    checkpoint_name: str,
    device: torch.device,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    assert checkpoint_name in MODEL_PATHS, f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_PATHS.keys()}."
    path = fetch_file_cached(
        MODEL_PATHS[checkpoint_name], progress=progress, 
        cache_dir=cache_dir, chunk_size=chunk_size
    )
    return torch.load(path, map_location=device)