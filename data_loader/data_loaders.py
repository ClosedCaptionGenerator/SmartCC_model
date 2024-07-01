import torch
import torch.utils
from torch.utils.data import DataLoader
from .dataset import MFCCDataset


def create_dataloaders(df, batch_size, sr, n_mfcc, n_fft, n_hop, transform=None):
    dataset = MFCCDataset(df, sr, n_mfcc, n_fft, n_hop, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
