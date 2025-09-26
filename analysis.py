import os
import json
import pandas as pd

base_dir = 'logs/num_finger_4'
results = []
for exp in os.listdir(base_dir):
    reward_path = os.path.join(base_dir,exp,'reward.json')
    if not os.path.exists(reward_path): continue
    with open(reward_path) as f:
        reward = json.load(f)['reward']
    
    [_,obj_name,quat,scale] = exp.split('_a_')
    quat_w,quat_x,quat_y,quat_z = quat.split('_')
    quat = [float(quat_w),float(quat_x),float(quat_y),float(quat_z)]
    scale = float(scale)
    results.append({
        "object": obj_name,
        "quat_w": quat_w,
        "quat_x": quat_x,
        "quat_y": quat_y,
        "quat_z": quat_z,
        "scale": scale,
        "reward": reward
    })

df = pd.DataFrame(results)
df.to_csv(os.path.join('logs',"experiment_results_4.csv"), index=False)

print("✅ Results saved to experiment_results.csv")
