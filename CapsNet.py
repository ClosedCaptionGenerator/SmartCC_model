import os

import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import lr_scheduler
from sklearn.preprocessing import OneHotEncoder


class CapsLayer(nn.Module):
    def __init__(self, num_capsule, dim_capsule, routings, in_channels):
        super(CapsLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.weights = nn.Parameter(torch.randn(1, in_channels, num_capsule, dim_capsule, in_channels))

    def squash(self, x):
        s_squared_norm = x.pow(2).sum(-1, keepdim=True)
        scale = s_squared_norm / (1 + s_squared_norm)
        return scale * x / torch.sqrt(s_squared_norm)

    def forward(self, x):
        batch_size = x.size(0)
        x_hat = torch.matmul(self.weights, x[:, None, :, :, None]).squeeze(-1)
        x_hat_detached = x_hat.detach()

        b = Variable(torch.zeros(batch_size, x.size(1), self.num_capsule, 1)).to(x.device)

        for i in range(self.routings):
            c = F.softmax(b, dim=2)
            if i == self.routings - 1:
                s = (c * x_hat).sum(dim=1, keepdim=True)
            else:
                s = (c * x_hat_detached).sum(dim=1, keepdim=True)
            v = self.squash(s)
            if i < self.routings - 1:
                b = b + (x_hat_detached * v).sum(dim=1, keepdim=True)

        return v.squeeze(1)


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, dim_capsule, n_channels, kernel_size, stride, padding):
        super(PrimaryCapsLayer, self).__init__()
        self.dim_capsule = dim_capsule
        self.capsules = nn.ModuleList(
            [nn.Conv2d(input_channels, n_channels, kernel_size=kernel_size, stride=stride, padding=padding)
             for _ in range(dim_capsule)]
        )

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.dim_capsule, -1)
        return u.permute(0, 2, 1)


class CapsNet(nn.Module):
    def __init__(self, input_shape, n_class, routings):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=(5, 5), stride=(2, 3),
                               padding='same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=256, kernel_size=(5, 5), stride=(1, 2), padding='valid')
        self.primaryCaps = PrimaryCapsLayer(input_channels=256, dim_capsule=8, n_channels=32, kernel_size=9, stride=2,
                                            padding='valid')
        self.digitCaps = CapsLayer(num_capsule=n_class, dim_capsule=16, routings=routings, in_channels=32 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.Linear(16 * n_class, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(np.prod(input_shape))),
            nn.Sigmoid(),
            nn.Unflatten(1, input_shape),
        )
        self.input_shape = input_shape

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        out_caps = x.pow(2).sum(dim=2).sqrt()
        masked_by_y = Mask()(x, y)
        masked = Mask()(x)
        recon_by_y = self.decoder(masked_by_y)
        recon = self.decoder(masked)
        return out_caps, recon, recon_by_y


class Mask(nn.Module):
    def forward(self, x, y=None):
        if y is None:
            y = torch.sqrt((x ** 2).sum(dim=2)).max(dim=1)[1]
            y = torch.eye(x.size(1)).to(x.device).index_select(dim=0, index=y)

        return (x * y[:, :, None]).view(x.size(0), -1)


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


class HDF5Dataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = []
        self.labels = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.h5'):
                    file_path = os.path.join(root, file)
                    with h5py.File(file_path, 'r') as h5_file:
                        self.file_paths.append(file_path)
                        self.labels.append(np.array(h5_file['y']))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], 'r') as h5_file:
            x_data = np.array(h5_file['mfcc'])
            y_data = np.array(h5_file['y'])
            return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.long)


# class MFCCDataset(Dataset):
#     def __init__(self, data_dir, labels, height, width, n_label, train=True):
#         self.data = []
#         self.labels = []
#         self.height = height
#         self.width = width
#
#         for i in range(n_label):
#             mfcc_file = f'{data_dir}/mfcc_y_{height}x{width}_{i}.h5'
#             with h5py.File(mfcc_file, 'r') as f:
#                 self.data.extend(f['mfcc'])
#                 self.labels.extend(f['y'])
#
#         self.data = np.array(self.data).reshape(-1, 1, height, width)
#         self.labels = np.array(self.labels)
#         self.labels = np.argmax(self.labels, axis=1)
#
#         encoder = OneHotEncoder(sparse_output=False)
#         self.labels = encoder.fit_transform(self.labels[:, None])
#
#         self.data = torch.tensor(self.data, dtype=torch.float32)
#         self.labels = torch.tensor(self.labels, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]


def load_dataset(train_data_dir, valid_data_dir, batch_size=32):
    train_dataset = HDF5Dataset(train_data_dir)
    valid_dataset = HDF5Dataset(valid_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet on Dataset.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--height', type=int, default=28,
                        help='input image height (default: 28)')
    parser.add_argument('--width', type=int, default=28,
                        help='input image width (default: 28)')
    parser.add_argument('--n_label', type=int, default=24,
                        help='number of classes (default: 24)')
    parser.add_argument('--train_dir', type=str, default='./data/train/raw',
                        help='Directory for storing training data (default: ./data/train/raw)')
    parser.add_argument('--valid_dir', type=str, default='./data/valid/raw',
                        help='Directory for storing validation data (default: ./data/val/raw)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='learning rate decay (default: 0.9)')
    parser.add_argument('--lam_reconstruction', type=float, default=0.392,
                        help='lambda for reconstruction loss (default: 0.392)')
    parser.add_argument('-r', '--routings', type=int, default=3,
                        help='number of routing iterations should > 0 (default: 3)')
    parser.add_argument('--shift_fraction', type=float, default=0.1,
                        help='fraction of input data to shift each batch (default: 0.1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints (default: ./checkpoints)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    train_loader, val_loader = load_dataset(args.train_dir, args.valid_dir, args.test_batch_size)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(f'Batch {batch_idx+1}:')
        print(f'Data shape: {data.shape}')
        print(f'Target shape: {target.shape}')

    # define model
    model = CapsNet(
        input_shape=(1, args.height, args.width),
        n_class=args.n_label,
        routings=args.routing_iterations
    )

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(16, args.n_label)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
    loss_fn = MarginLoss(0.9, 0.1, 0.5)


    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()

            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in val_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784), size_average=False).data[0]
                test_loss += loss_fn(probs, target, size_average=False).data[0]
                test_loss += reconstruction_alpha * reconstruction_loss
            else:
                output, probs = model(data)
                test_loss += loss_fn(probs, target, size_average=False).data[0]

            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)\n')

        return test_loss


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test()
        scheduler.step(test_loss)
        torch.save(model.state_dict(),
                   f'{epoch:03d}_model_dict_{args.routing_iterations}routing_reconstruction{args.with_reconstruction}.pth')
