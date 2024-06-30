import torch
import torch.nn as nn
import torch.nn.functional as F
from .capsule_layers import PrimaryCapsules, DigitCapsules

class CapsuleNet(nn.Module):
    def __init__(self, num_classes=24):  # Update the number of classes to 24
        super(CapsuleNet, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=19, stride=1)
        self.primary_capsules = PrimaryCapsules(num_capsules=32, in_channels=256, out_channels=8, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsules(num_capsules=num_classes, num_routes=32*6*6, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16*num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv_layer(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        reconstructions = self.decoder(x.view(x.size(0), -1))
        return x, reconstructions

def capsule_net(num_classes=24):
    return CapsuleNet(num_classes)
