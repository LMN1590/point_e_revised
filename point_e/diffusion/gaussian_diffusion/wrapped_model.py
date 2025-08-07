import torch as th
import torch.nn as nn

from typing import List
class _WrappedModel:
    def __init__(self, model:nn.Module, timestep_map:List[int], original_num_steps:int):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps

    def __call__(self, x:th.Tensor, ts:th.Tensor, **kwargs):
        kwargs['original_ts'] = ts
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs)