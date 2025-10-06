from utils import init_log_dir
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
ctrl_tensor = torch.tensor([[0.5,0.3,0.3,0.75,0.3,0.,0.,0.,0.,0.,0.]])
ctrl_tensor = ctrl_tensor.repeat(4,4,1)
ctrl_tensor.requires_grad_(True)
gripper = designer._transform_splining(4,4,ctrl_tensor)
full_gripper, end_pts = designer._transform_lengthening(4,4,ctrl_tensor,gripper)
print(full_gripper)
print(full_gripper[0,0].max(0),full_gripper[0,0].min(0))


segment = full_gripper[0,0].detach().cpu().numpy()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(segment)

# Optional: set color
pcd.paint_uniform_color([0.2, 0.7, 1.0])

# Visualize
o3d.visualization.draw_geometries([pcd])