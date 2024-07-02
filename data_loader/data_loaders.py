import torch
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
from .dataset import MFCCDataset


def create_dataloaders(df_train, df_val, batch_size, sr, n_mfcc, n_fft, n_hop, max_len, width, transform=None):
    train_dataset = MFCCDataset(df_train, sr, n_mfcc, n_fft, n_hop, max_len, width, transform=transform)
    valid_dataset = MFCCDataset(df_val, sr, n_mfcc, n_fft, n_hop, max_len, width, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    sample_data_x, sample_data_y = next(iter(train_loader))
    input_shape = sample_data_x.shape[1:]  # Exclude the batch size
    y_train = df_train['classID'].values
    n_class = len(np.unique(y_train))

    return train_loader, test_loader, input_shape, n_class
