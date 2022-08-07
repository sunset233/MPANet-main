import torch
import torch.nn.functional as F

def gemp(x):
    p = 3.0
    return (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)

a = torch.randn(32, 2048, 18, 9)
b = a.view(a.size(0), a.size(1), -1)
y = gemp(b)
print(y.shape)

z = F.avg_pool2d(a, a.size()[2:])
z = z.view(z.size(0), -1)
print(z.shape)