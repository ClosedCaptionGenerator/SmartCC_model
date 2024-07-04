import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-9)



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCap(nn.Module):
    def __init__(self, in_channels, dim_capsule, n_channels, kernel_size, stride, padding):
        super(PrimaryCap, self).__init__()
        self.conv = nn.Conv2d(in_channels, dim_capsule * n_channels, kernel_size, stride, padding)
        self.n_channels = n_channels
        self.dim_capsule = dim_capsule

    def forward(self, x):
        x = self.conv(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.dim_capsule)
        x = squash(x)
        return x


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsule, dim_capsule, routings=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.W = None

    def forward(self, inputs):
        device = inputs.device
        batch_size = inputs.size(0)
        input_num_capsule = inputs.size(1)
        input_dim_capsule = inputs.size(2)

        if self.W is None:
            self.W = nn.Parameter(torch.randn(1, self.num_capsule, input_num_capsule, self.dim_capsule, input_dim_capsule).to(device))

        inputs = inputs.unsqueeze(1).unsqueeze(4)
        inputs = inputs.repeat(1, self.num_capsule, 1, 1, 1)
        inputs_hat = torch.matmul(self.W, inputs)

        b = torch.zeros_like(inputs_hat[:, :, :, :, 0]).to(device)

        for i in range(self.routings):
            c = F.softmax(b, dim=2)
            outputs = squash((c.unsqueeze(4) * inputs_hat).sum(dim=2, keepdim=True))

            if i < self.routings - 1:
                b = b + (inputs_hat * outputs).sum(dim=-1)

        return outputs.squeeze(2)


class Decoder(nn.Module):
    def __init__(self, input_shape, n_class):
        super(Decoder, self).__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            nn.Linear(n_class * 16, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, int(torch.prod(torch.tensor(input_shape)))),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = x.view(x.size(0), *self.input_shape)
        return x


class Distance(nn.Module):
    def forward(self, inputs):
        return torch.sqrt(torch.sum(inputs ** 2, dim=-1))


class Mask(nn.Module):
    def forward(self, inputs, mask=None):
        if mask is None:
            lengths = torch.sqrt((inputs ** 2).sum(dim=-1))
            _, max_length_indices = lengths.max(dim=1)
            mask = torch.eye(inputs.size(1)).to(inputs.device).index_select(dim=0, index=max_length_indices)
        else:
            mask = mask

        masked = inputs * mask.unsqueeze(-1)
        masked = masked.view(masked.size(0), -1)
        return masked


class CapsNet(nn.Module):
    def __init__(self, input_shape, n_class, config):
        super(CapsNet, self).__init__()
        self.n_label = config['n_label']
        self.conv1 = ConvLayer(
            config['cnn1_in_channels'],
            config['cnn1_out_channels'],
            tuple(config['cnn1_kernel_size']),
            tuple(config['cnn1_stride']),
            tuple(config['cnn1_padding'])
        )
        self.conv2 = ConvLayer(
            config['cnn2_in_channels'],
            config['cnn2_out_channels'],
            tuple(config['cnn2_kernel_size']),
            tuple(config['cnn2_stride']),
            tuple(config['cnn2_padding'])
        )
        self.primary_capsules = PrimaryCap(
            config['pc_in_channels'],
            config['pc_dim_capsule'],
            config['pc_n_channels'],
            config['pc_kernel_size'],
            config['pc_stride'],
            config['pc_padding']
        )
        self.digit_capsules = CapsuleLayer(
            n_class,
            config['dc_dim_capsule'],
            config['routings']
        )
        self.distance = Distance()
        self.mask = Mask()
        self.decoder = Decoder(input_shape=input_shape, n_class=n_class)

    def forward(self, x, y=None):
        device = x.device  # Ensure all tensors are on the same device
        x = self.conv1(x.to(device))
        x = self.conv2(x)
        x = self.primary_capsules(x)
        digitcaps_output = self.digit_capsules(x)
        out_caps = self.distance(digitcaps_output)

        if y is not None:
            y = y.to(device)
            masked_output = self.mask(out_caps, y)
        else:
            masked_output = self.mask(out_caps)

        reconstructions = self.decoder(masked_output)
        return out_caps, reconstructions


# # Load config
# import json
# with open('./config/config.json', 'r') as f:
#     config = json.load(f)

# # Instantiate and verify the shape
# input_shape = (1, 48, 173)
# capsnet = CapsNet(input_shape=input_shape, n_class=config['n_label'], config=config)
# dummy_input = torch.randn(256, *input_shape)
# out_caps, recon = capsnet(dummy_input)

# print(f"Final output capsule shape: {out_caps.shape}")
# print(f"Reconstructed image shape: {recon.shape}")

