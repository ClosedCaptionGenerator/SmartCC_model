import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Adjust dimensions of W to match expected input shape for matmul
        self.W = nn.Parameter(torch.randn(1, num_capsules, num_routes, out_channels, in_channels))

    def forward(self, x):
        # x shape: (batch_size, num_routes, in_channels)
        batch_size = x.size(0)

        # Reshape the input to match the expected dimensions
        x = x.view(batch_size, self.num_routes, self.in_channels)

        # Expand input for batch processing and to match weight dimensions
        x = x.unsqueeze(1).unsqueeze(-1)

        # Apply weight matrix
        W = self.W.repeat(batch_size, 1, 1, 1, 1)

        u_hat = torch.matmul(W, x).squeeze(-1)

        # Squash function
        v = self.squash(u_hat)

        return v

    def squash(self, inputs):
        sq_norm = (inputs ** 2).sum(dim=-1, keepdim=True)
        scale = sq_norm / (1 + sq_norm)
        return scale * inputs / torch.sqrt(sq_norm + 1e-9)
