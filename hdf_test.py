import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class HDF5TestDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(hdf5_file, 'r') as file:
            self.data = file['data'][:]
            self.labels = file['labels'][:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def get_test_loader(hdf5_file, batch_size):
    dataset = HDF5TestDataset(hdf5_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
