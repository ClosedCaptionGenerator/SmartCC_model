import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCapsules(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size, stride) for _ in range(num_capsules)]
        )

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), -1, u.size(-2), u.size(-1))
        return u

class DigitCapsules(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels):
        super(DigitCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_routes, in_channels, out_channels))

    def forward(self, x):
        x = x.unsqueeze(2).expand(x.size(0), x.size(1), self.num_routes, x.size(3)).contiguous()
        x = x.view(x.size(0), self.num_capsules, -1)
        x = torch.matmul(x, self.route_weights)
        x = x.sum(dim=1)
        return x
