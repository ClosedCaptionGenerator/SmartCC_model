import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .capsule_layers import PrimaryCapsule, CapsuleLayer


class CapsuleNet(nn.Module):
    def __init__(self, input_shape, n_class, routings):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=(2, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=256, kernel_size=5, stride=(1, 2), padding='valid')

        self.primary_capsules = PrimaryCapsule(dim_capsule=8, n_channels=32, kernel_size=9, stride=2, padding='valid')
        self.digit_capsules = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings)

        self.decoder = nn.Sequential(
            nn.Linear(16 * n_class, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, np.prod(input_shape)),
            nn.Sigmoid()
        )

        self.input_shape = input_shape

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        primary_caps_output = self.primary_capsules(x)
        digit_caps_output = self.digit_capsules(primary_caps_output)
        out_caps = torch.sqrt(torch.sum(digit_caps_output ** 2, dim=-1) + 1e-9)

        if y is None:
            mask = torch.argmax(out_caps, dim=1)
        else:
            mask = y.float()

        masked = digit_caps_output * mask.unsqueeze(2)
        masked = masked.view(masked.size(0), -1)

        reconstruction = self.decoder(masked)
        reconstruction = reconstruction.view(-1, *self.input_shape)

        return out_caps, reconstruction


def margin_loss(y_true, y_pred):
    left = F.relu(0.9 - y_pred).pow(2)
    right = F.relu(y_pred - 0.1).pow(2)
    loss = y_true * left + 0.5 * (1.0 - y_true) * right
    return loss.mean()
