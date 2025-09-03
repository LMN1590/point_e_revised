import torch
import kaolin.ops.mesh

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Input tensors (from your Kaolin example)
verts = torch.tensor([[[0., 0., 0.],
                       [1., 0.5, 1.],
                       [0.5, 1., 1.],
                       [1., 1., 0.5]]], device=device)
faces = torch.tensor([[0, 3, 1],
                      [0, 1, 2],
                      [0, 2, 3],
                      [3, 2, 1]], device=device)
axis = torch.linspace(0.1, 0.9, 3, device=device)
p_x, p_y, p_z = torch.meshgrid(axis + 0.01, axis + 0.02, axis + 0.03)
points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
pts_gt = points.view(1, -1, 3)  # Matches your `pts_gt` variable

# Compute occupancy using Kaolin's check_sign
occ = kaolin.ops.mesh.check_sign(verts, faces, pts_gt)
print(verts.shape,faces.shape,pts_gt.shape,occ.shape)

# Apply weights (from your Open3D snippet)
x_0_weight = torch.ones((1, pts_gt.shape[1]), device=device)  # Example weight tensor
cfg = {'train': {'w_inside': 2.0, 'w_outside': 1.0}}  # Example config
x_0_weight[occ] *= cfg['train']['w_inside']  # Weight for inside points
x_0_weight[~occ] *= cfg['train']['w_outside']  # Weight for outside points

print(x_0_weight.shape)