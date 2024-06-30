import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class SoundDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = self._get_data_files()

    def _get_data_files(self):
        data_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.raw'):
                    data_files.append(os.path.join(root, file))
        return data_files

    def _load_data(self, data_path):
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        return data

    def _load_label(self, label_path):
        with open(label_path, 'r') as f:
            label = json.load(f)
        return label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        label_path = data_path.replace('raw', 'label').replace('.raw', '.json')

        data = self._load_data(data_path)
        label = self._load_label(label_path)

        if self.transform:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_dir, val_dir, batch_size):
    train_dataset = SoundDataset(train_dir)
    val_dataset = SoundDataset(val_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
