import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def squash(tensor, dim=-1):
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    norm_squared = norm ** 2
    return (norm_squared / (1 + norm_squared)) * (tensor / norm)

class Distance(nn.Module):
    def forward(self, inputs):
        return torch.sqrt(torch.sum(inputs ** 2, dim=-1))

class Mask(nn.Module):
    def forward(self, inputs, mask=None):
        if mask is None:
            lengths = torch.sqrt((inputs ** 2).sum(dim=-1))
            mask = F.one_hot(torch.argmax(lengths, dim=1), num_classes=inputs.size(1)).float()
        else:
            mask = mask

        masked = inputs * mask[:, :, None]
        return masked.view(masked.size(0), -1)

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
        # x = x.view(batch_size, -1, x.size(1) // self.n_channels)
        x = x.view(batch_size, -1, self.dim_capsule)
        x = self.squash(x)
        return x

    def squash(self, x):
        squared_norm = (x ** 2).sum(dim=2, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)



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

        inputs = inputs.unsqueeze(2).unsqueeze(4)
        inputs = inputs.repeat(1, 1, self.num_capsule, 1, 1)
        inputs = torch.matmul(self.W, inputs)

        b = torch.zeros_like(inputs[:, :, :, :, 0])

        for i in range(self.routings):
            c = F.softmax(b, dim=2)
            outputs = self.squash((c.unsqueeze(4) * inputs).sum(dim=1, keepdim=True))

            if i < self.routings - 1:
                b = b + (inputs * outputs).sum(dim=-1)

        return outputs.squeeze(1)

        # if self.W is None:
        #     self.W = nn.Parameter(torch.randn(input_num_capsule, self.num_capsule, input_dim_capsule, self.dim_capsule).to(device))

        # inputs = inputs[:, :, None, :, None]
        # W = self.W[None, :, :, :, :]

        # u_hat = torch.matmul(W, inputs).squeeze(-1)
        # b = torch.zeros(batch_size, input_num_capsule, self.num_capsule, 1).to(device)

        # for i in range(self.routings):
        #     c = F.softmax(b, dim=2)  # Compute routing coefficients
        #     s = (c * u_hat).sum(dim=1, keepdim=True)  # Weighted sum of u_hat
        #     v = self.squash(s)  # Apply squash function

        #     if i < self.routings - 1:
        #         b = b + (u_hat * v).sum(dim=-1, keepdim=True)

        # return v.squeeze(1)

class Decoder(nn.Module):
    def __init__(self, input_shape, n_class):
        super(Decoder, self).__init__()
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
        x = self.decoder(x)
        x = x.view(x.size(0), *self.input_shape)
        return x

class CapsNet(nn.Module):
    def __init__(self, input_shape, n_class, config):
        super(CapsNet, self).__init__()
        self.n_label = config['n_label']
        self.conv1 = ConvLayer(config['cnn1_in_channels'], config['cnn1_out_channels'], tuple(config['cnn1_kernel_size']), tuple(config['cnn1_stride']), tuple(config['cnn1_padding']))
        self.conv2 = ConvLayer(config['cnn2_in_channels'], config['cnn2_out_channels'], tuple(config['cnn2_kernel_size']), tuple(config['cnn2_stride']), config['cnn2_padding'])
        self.primary_capsules = PrimaryCap(config['pc_in_channels'], config['pc_dim_capsule'], config['pc_n_channels'], config['pc_kernel_size'], config['pc_stride'], config['pc_padding'])
        self.digit_capsules = CapsuleLayer(n_class, config['dc_dim_capsule'], config['routings'])
        self.distance = Distance()
        self.mask = Mask()
        self.decoder = Decoder(input_shape=input_shape, n_class=n_class)

    def forward(self, x, y=None):
        device = x.device  # Ensure all tensors are on the same device
        print(f'x shape 1: {x.shape}')
        x = self.conv1(x.to(device))
        print(f'x shape 2: {x.shape}')
        x = self.conv2(x)
        print(f'x shape 3: {x.shape}')
        x = self.primary_capsules(x)
        print(f'x shape 4: {x.shape}')
        digitcaps_output = self.digitcaps(x)
        print(f'x shape 4: {digitcaps_output.shape}')
        out_caps = self.distance(digitcaps_output)
        print(f'x shape 6: {out_caps.shape}')

        if y is not None:
            y = y.to(device)
            masked_output = self.mask(digitcaps_output, y)
        else:
            masked_output = self.mask(digitcaps_output)

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
