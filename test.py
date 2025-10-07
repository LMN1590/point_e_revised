from utils import init_log_dir
import numpy as np
import json
import torch
import open3d as o3d
init_log_dir('debug','test','debug/tensorboard',increment_step=1.)

from diff_conditioning.simulation_env.designer.encoded_finger.design import EncodedFinger
with open("diff_conditioning/simulation_env/designer/encoded_finger/config/base_config.json") as f:
    base_config = json.load(f)
designer = EncodedFinger(
    base_config=base_config
)
ctrl_tensor = torch.tensor([[0.25,0.3,0.3,0.75,0.3,1.,0.,0.,0.05,0.,0.]])
ctrl_tensor = ctrl_tensor.repeat(4,4,1)
ctrl_tensor.requires_grad_(True)
gripper = designer._create_gripper(ctrl_tensor)



segment = gripper[0].detach().cpu().numpy()
print(segment.max(0),segment.min(0))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(segment)

colors = np.vstack([
    np.zeros((666, 3)),
    np.tile(np.array([1.,0.,0.]), (666, 1)),
    np.tile(np.array([0.,1.,0.]), (666, 1)),
    np.tile(np.array([0.,0.,1.]), (666, 1))
])
pcd.colors = o3d.utility.Vector3dVector(colors)

# Optional: set color
# pcd.paint_uniform_color([0.2, 0.7, 1.0])

# Visualize
o3d.visualization.draw_geometries([pcd])