# capsnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False

def squash(vectors, dim=-1):
    s_squared_norm = torch.sum(vectors ** 2, dim=dim, keepdim=True)
    scale = s_squared_norm / (1 + s_squared_norm) / torch.sqrt(s_squared_norm + 1e-9)
    return scale * vectors


class Distance(nn.Module):
    def forward(self, inputs):
        return torch.sqrt(torch.sum(inputs ** 2, dim=-1) + 1e-9)


class Mask(nn.Module):
    def forward(self, inputs, mask=None):
        if mask is None:  # No true label provided, mask by the max length of capsules
            lengths = torch.sqrt((inputs ** 2).sum(dim=-1))
            mask = F.one_hot(torch.argmax(lengths, dim=1), num_classes=inputs.size(1)).float()
        else:  # True label provided
            mask = mask.float()

        masked = inputs * mask.unsqueeze(-1)
        return masked.view(masked.size(0), -1)


def calculate_same_padding(input_height, input_width, kernel_height, kernel_width, stride_height, stride_width):
    if (input_height % stride_height == 0):
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (input_height % stride_height), 0)

    if (input_width % stride_width == 0):
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (input_width % stride_width), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, dim_capsule, n_channels, kernel_size, stride, padding):
        super(PrimaryCaps, self).__init__()
        self.dim_capsule = dim_capsule
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=dim_capsule * n_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1, self.dim_capsule)
        x = squash(x)
        return x


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsule, dim_capsule, routings=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.W = nn.Parameter(torch.randn(1, num_capsule, dim_capsule, 8))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).unsqueeze(4)
        x = x.permute(0, 1, 3, 4, 2)
        u_hat = torch.matmul(self.W, x)
        b = torch.zeros(batch_size, self.num_capsule, x.size(2), 1).to(x.device)

        for i in range(self.routings):
            c = torch.nn.functional.softmax(b, dim=1)
            s = (c * u_hat).sum(dim=2, keepdim=True)
            v = squash(s)
            if i < self.routings - 1:
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)

        return v.squeeze(3)


class Decoder(nn.Module):
    def __init__(self, input_shape, n_class):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(16 * n_class, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, np.prod(input_shape))
        self.input_shape = input_shape

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, *self.input_shape)


class CapsNet(nn.Module):
    def __init__(self, input_shape, config):
        super(CapsNet, self).__init__()
        self.n_label = config['n_label']
        self.conv1 = ConvLayer(config['cnn1_in_channels'], config['cnn1_out_channels'], config['cnn1_kernel_size'], config['cnn1_kernel_size'], config['cnn1_padding'])
        self.conv2 = ConvLayer(config['cnn2_in_channels'], config['cnn2_out_channels'], config['cnn2_kernel_size'], config['cnn2_kernel_size'], config['cnn2_padding'])
        self.primary_capsules = PrimaryCaps(config['pc_in_channels'], config['pc_dim_capsule'], config['pc_n_channels'], config['pc_kernel_size'], config['pc_stride'], config['pc_padding'])
        self.digit_capsules = CapsuleLayer(self.n_label, config['dc_dim_capsule'], config['routings'])
        self.distance = Distance()
        self.mask = Mask()
        self.decoder = Decoder(input_shape=input_shape, n_class=self.n_label)

    def forward(self, x, y=None):
        print(f"Shape before conv1: {x.shape}")
        x = self.conv1(x)
        print(f"Shape after conv1: {x.shape}")
        x = self.conv2(x)
        print(f"Shape after conv2: {x.shape}")
        x = self.primary_capsules(x)
        print(f"Shape after primary capsules: {x.shape}")
        x = self.digit_capsules(x)
        print(f"Shape after digit capsules: {x.shape}")
        out_caps = self.distance(x)

        if y is None:
            y = torch.eye().index_select(dim=0, index=out_caps.argmax(dim=1)).to(x.device)

        masked = self.mask(x, y)
        reconstruction = self.decoder(masked)
        return out_caps, reconstruction

import json
# Instantiate and verify the shape
input_shape = (1, 48, 173)
with open('./config/config.json', 'r') as f:
    config = json.load(f)
capsnet = CapsNet(input_shape, config)
dummy_input = torch.randn(256, *input_shape)
out_caps, recon = capsnet(dummy_input)

print(f"Final output capsule shape: {out_caps.shape}")
print(f"Reconstructed image shape: {recon.shape}")
