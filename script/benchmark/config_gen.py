import os
import yaml
import copy
import numpy as np
from itertools import product
from tqdm import tqdm
import random
from pyquaternion import Quaternion

def generate_quaternion_grid(resolution=2):
    """
    Generate all possible quaternions from discretized Euler angles using pyquaternion.
    
    Parameters:
        resolution (int): number of discrete values per axis.
                          e.g. 5 means angles = [0, 90, 180, 270, 360).
    
    Returns:
        quaternions (np.ndarray): shape (resolution^3, 4), each row is (w, x, y, z).
    """
    # Create discretized angles in radians (exclude 360=2π to avoid duplicates)
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    all_combos = product(angles, repeat=3)  # (roll, pitch, yaw)

    quaternions = []
    for combo in all_combos:
        q = Quaternion(axis=[1,0,0], angle=combo[0]) \
          * Quaternion(axis=[0,1,0], angle=combo[1]) \
          * Quaternion(axis=[0,0,1], angle=combo[2])
        quaternions.append([q.w.item(), q.x.item(), q.y.item(), q.z.item()])
    
    return quaternions

if __name__ == "__main__":
    base_config_path = 'softzoo/configs/env_configs/benchmark_lifting.yaml'
    benchmark_config_path = 'script/benchmark/benchmark_config'
    os.makedirs(benchmark_config_path,exist_ok=True)
    quats = generate_quaternion_grid(4)
    real_scales = [0.3,0.5,0.7]
    with open(base_config_path) as f:
        env_config = yaml.safe_load(f)

    base_dir = '/media/aioz-nghiale/data1/Data/mujoco_scanned_objects/models'
    for obj in tqdm(os.listdir(base_dir)[:5]):
        obj_path = os.path.join(base_dir,obj,'model.obj')
        random_quats = random.sample(quats,k=4)
        for quat in random_quats:
            for real_scale in real_scales:
                base_config = copy.deepcopy(env_config)
                mesh_config = [item for item in base_config['ENVIRONMENT']['ITEMS'] if item['type']=='Primitive.Mesh'][0]
                mesh_config['file_path'] = obj_path
                mesh_config['scale'] = [real_scale for _ in range(3)]
                mesh_config['initial_rotation'] = quat
                
                config_name = f'gripping_a_{obj}_a_{"_".join([str(item)for item in quat])}_a_{real_scale}.yaml'
                with open(os.path.join(benchmark_config_path,config_name),'w') as f:
                    yaml.safe_dump(base_config,f)
    # print(count)