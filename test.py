import torch

# Example points (requires_grad=True so we can backprop)
points = torch.tensor([
    [0.25, 0.25, 0.25],
    [0.4, 0.25, 0.25]
], requires_grad=True)

# Example voxel centers
voxel_coords = torch.stack(torch.meshgrid(
    torch.linspace(0., 1., 4),
    torch.linspace(0., 1., 4),
    torch.linspace(0., 1., 4),
), dim=-1).reshape(-1, 3)  # (m,3)

# Kernel sharpness
sigma = 0.1  # ~softness, fraction of voxel size

# Distance from each voxel center to each point
dist = torch.cdist(voxel_coords, points)  # (m, n)

# Smooth per-point contribution (Gaussian kernel)
p_ji = torch.exp(-(dist**2) / (2 * sigma**2))  # (m, n)

# Soft OR across points → final occupancy per voxel
occupancy = 1 - torch.prod(1 - p_ji, dim=1)  # (m,)

# Example loss: maximize occupancy at voxel 10
loss = -occupancy[10]
loss.backward()

print(voxel_coords[10])
print("Occupancy at voxel 10:", occupancy[10].item())
print("Gradients wrt points:\n", points.grad)
occ = torch.concat([voxel_coords,occupancy.unsqueeze(1)],dim=-1)
print(occ)

print(points.min(0).values - points.max(0).values)