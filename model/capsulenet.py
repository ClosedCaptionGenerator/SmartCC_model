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
        return torch.sqrt(torch.sum(inputs ** 2, dim=-1) + 1e-9)

class Mask(nn.Module):
    def forward(self, inputs, mask=None):
        if mask is None:
            lengths = torch.sqrt((inputs ** 2).sum(dim=-1))
            mask = F.one_hot(torch.argmax(lengths, dim=1), num_classes=inputs.size(1)).float()
        else:
            mask = mask.float()

        masked = inputs * mask.unsqueeze(-1)
        masked = masked.view(masked.size(0), -1)
        return masked

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
        self.n_channels = n_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=dim_capsule * n_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1, self.dim_capsule)
        x = squash(x)
        return x

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsule, dim_capsule, batch_size, routings=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.batch_size = batch_size
        self.routings = routings
        self.W = None

    def forward(self, inputs):
        device = inputs.device
        input_num_capsule = inputs.size(1)
        input_dim_capsule = inputs.size(2)

        if self.W is None:
            self.W = nn.Parameter(torch.randn(1, self.num_capsule, input_num_capsule, self.dim_capsule, input_dim_capsule).to(device))

        inputs_expand = inputs.unsqueeze(1).unsqueeze(4)
        inputs_tiled = inputs_expand.repeat(1, self.num_capsule, 1, 1, 1)
        W_tiled = self.W.repeat(inputs.size(0), 1, 1, 1, 1)  # [batch_size, num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        inputs_hat = torch.matmul(W_tiled, inputs_tiled).squeeze(-1)  # [batch_size, num_capsule, input_num_capsule, dim_capsule]

        b_ij = torch.zeros(self.batch_size, self.num_capsule, input_num_capsule).to(device)

        for i in range(self.routings):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = c_ij.unsqueeze(3)
            s_j = (c_ij * inputs_hat).sum(dim=2)
            v_j = self.squash(s_j)

            if i < self.routings - 1:
                a_ij = torch.einsum('bijt,bijt->bij', inputs_hat, v_j.unsqueeze(2))
                b_ij = b_ij + a_ij

        return v_j

    def squash(self, inputs, axis=-1):
        squared_norm = (inputs ** 2).sum(axis, keepdim=True)
        scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-9)
        return scale * inputs


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
    def __init__(self, input_shape, n_class, config):
        super(CapsNet, self).__init__()
        self.n_label = config['n_label']
        self.conv1 = ConvLayer(config['cnn1_in_channels'], config['cnn1_out_channels'], tuple(config['cnn1_kernel_size']), tuple(config['cnn1_stride']), tuple(config['cnn1_padding']))
        self.conv2 = ConvLayer(config['cnn2_in_channels'], config['cnn2_out_channels'], tuple(config['cnn2_kernel_size']), tuple(config['cnn2_stride']), config['cnn2_padding'])
        self.primary_capsules = PrimaryCaps(config['pc_in_channels'], config['pc_dim_capsule'], config['pc_n_channels'], config['pc_kernel_size'], config['pc_stride'], config['pc_padding'])
        self.digit_capsules = CapsuleLayer(n_class, config['dc_dim_capsule'], config['batch_size'], config['routings'])
        self.distance = Distance()
        self.mask = Mask()
        self.decoder = Decoder(input_shape=input_shape, n_class=n_class)

    def forward(self, x, y=None):
        device = x.device  # Ensure all tensors are on the same device
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        out_caps = self.distance(x)

        if y is None:
            y = torch.eye(self.n_label, device=device).index_select(dim=0, index=out_caps.argmax(dim=1)).to(device)

        masked = self.mask(x, y)
        reconstruction = self.decoder(masked)
        return out_caps, reconstruction

    def margin_loss(self, y_true, y_pred):
        print(f'y_true shape: {y_true.shape}')
        print(f'y_pred shape: {y_pred.shape}')
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5

        v_c = torch.sqrt((y_pred ** 2).sum(dim=-1, keepdim=True))  # Calculate the length of each capsule vector
        zero = torch.zeros(1, device=y_true.device)

        L_c = y_true * torch.max(zero, m_plus - v_c).pow(2) + \
              lambda_val * (1 - y_true) * torch.max(zero, v_c - m_minus).pow(2)

        return L_c.sum(dim=1).mean()

    def reconstruction_loss(self, x, reconstructions):
        return F.mse_loss(reconstructions, x)

    def combined_loss(self, x, y_pred, y_true, reconstructions, lam_recon):
        return self.margin_loss(y_true, y_pred) + lam_recon * self.reconstruction_loss(x, reconstructions)
