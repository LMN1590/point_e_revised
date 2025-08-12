import os
import json
from typing import Dict,List


def read_fixed_velocity(velocity_file:str,max_frames:int):
    velocity_path = os.path.join('./fixed_velocity',velocity_file)
    with open(velocity_path) as f:
        velocity:Dict[str,List[float]] = json.load(f)
    sorted_velocity_by_frame = sorted(velocity.items(),key=lambda x:int(x[0]))
    sorted_velocity_by_frame = list(map(lambda x:(int(x[0]),x[1]),sorted_velocity_by_frame))
    sorted_velocity_by_frame.append((max_frames+1,[0.,0.,0.]))
    return sorted_velocity_by_frame