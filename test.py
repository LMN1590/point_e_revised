import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

t = torch.linspace(0,0.1,6)
x = torch.tensor([0,4,16,36,64,100]).float().unsqueeze(1)
x.requires_grad_(True)
coeffs = natural_cubic_spline_coeffs(t, x)
spline = NaturalCubicSpline(coeffs)

point = torch.tensor(0.01)
out = spline.evaluate(point)
print(out)