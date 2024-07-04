import os
import numpy as np
import torch
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import glob

class MFCCDataset(Dataset):
    def __init__(self, mfcc_data, labels):
        self.mfcc_data = mfcc_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mfcc = self.mfcc_data[idx]
        label = self.labels[idx]
        mfcc = torch.tensor(mfcc, dtype=torch.float32).permute(2, 0, 1)  # Change the shape to [C, H, W]
        label = torch.tensor(label, dtype=torch.long)
        return mfcc, label


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, sr, n_mfcc, n_fft, n_hop, max_len, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio file
        audio, sr = librosa.load(file_path, sr=self.sr, duration=self.max_len)

        # Ensure the audio length is consistent
        if len(audio) < self.max_len * self.sr:
            padding = self.max_len * self.sr - len(audio)
            audio = np.pad(audio, (0, int(padding)), 'constant')

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.n_hop)
        pad_width = max(0, self.n_mfcc - mfcc.shape[1])
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

        if self.transform:
            mfcc = self.transform(mfcc)

        # Convert label to tensor
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0) 
        label = torch.tensor(label, dtype=torch.long)

        return mfcc, label


def get_file_paths_and_labels(data_dir, label_mapping):
    file_paths = []
    labels = []

    for label_name, label_id in label_mapping.items():
        # Adjust the pattern to match the nested directory structure
        pattern = os.path.join(data_dir, '**', label_name, '*.wav')
        class_files = glob.glob(pattern, recursive=True)
        print(f"Found {len(class_files)} files for label '{label_name}' with pattern {pattern}")
        file_paths.extend(class_files)
        labels.extend([label_id] * len(class_files))

    return file_paths, labels


def load_dataset(save_hdf_train, save_hdf_validation, n_label, height, width):
    x_train_mfcc = []
    y_train_mfcc = []
    x_val_mfcc = []
    y_val_mfcc = []

    for i in range(n_label):  # 0~23 labels
        for ds_name in ['mfcc', 'y']:
            if ds_name == 'mfcc':
                count = f"mfcc_y_{height}x{width}_{i}"
                with h5py.File(save_hdf_train + count + '.h5', 'r') as mfcc:
                    x_train_mfcc.extend(mfcc[ds_name])
            if ds_name == 'y':
                count = f"mfcc_y_{height}x{width}_{i}"
                with h5py.File(save_hdf_train + count + '.h5', 'r') as mfcc:
                    y_train_mfcc.extend(mfcc[ds_name])

        for ds_name in ['mfcc', 'y']:
            if ds_name == 'mfcc':
                count = f"mfcc_y_{height}x{width}_{i}"
                with h5py.File(save_hdf_validation + count + '.h5', 'r') as mfcc:
                    x_val_mfcc.extend(mfcc[ds_name])
            if ds_name == 'y':
                count = f"mfcc_y_{height}x{width}_{i}"
                with h5py.File(save_hdf_validation + count + '.h5', 'r') as mfcc:
                    y_val_mfcc.extend(mfcc[ds_name])

    train_x = np.array(x_train_mfcc).reshape(-1, height, width, 1)
    test_x = np.array(x_val_mfcc).reshape(-1, height, width, 1)
    train_y = np.array(y_train_mfcc)
    test_y = np.array(y_val_mfcc)

    train_y = np.argmax(train_y, axis=1).reshape(-1)
    test_y = np.argmax(test_y, axis=1).reshape(-1)
    train_y = F.one_hot(torch.tensor(train_y), num_classes=n_label).float()
    test_y = F.one_hot(torch.tensor(test_y), num_classes=n_label).float()

    print('normal x:', train_x.shape, test_x.shape)
    print('normal y:', train_y.shape, test_y.shape)

    return (train_x, train_y), (test_x, test_y)
