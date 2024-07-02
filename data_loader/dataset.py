import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


def id_parse(label_name):
    label_mapping = {
        '1.차량경적': 0, '2.차량사이렌': 1, '3.차량주행음': 2, '4.이륜차경적': 3,
        '5.이륜차주행음': 4, '6.비행기': 5, '7.헬리콥터': 6, '8.기차': 7,
        '9.지하철': 8, '10.발소리': 9, '11.가구소리': 10, '12.청소기': 11,
        '13.세탁기': 12, '14.개': 13, '15.고양이': 14, '16.공구': 15,
        '17.악기': 16, '18.항타기': 17, '19.파쇄기': 18, '20.콘크리트펌프': 19,
        '21.발전기': 20, '22.절삭기': 21, '23.송풍기': 22, '24.압축기': 23
    }
    return label_mapping.get(label_name, 24)

def extract_mfcc(file_path, sr, n_mfcc, n_fft, n_hop):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=n_hop)
        return mfcc
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def prepare_data(file_path_train):
    dirlist = []
    index = []
    filename = []

    for i in file_path_train:
        try:
            filename.append(i.split('/')[-1])
            dirlist.append(i)
            index.append(id_parse(i.split('/')[-2]))
        except Exception as e:
            print(e)

    df = pd.DataFrame({'classID': index, 'dir_filelist': dirlist, 'slice_filename': filename})
    df['classID'] = df['classID'].astype(int)
    return df

class MFCCDataset(Dataset):
    def __init__(self, df, sr, n_mfcc, n_fft, n_hop, max_len, width, transform=None):
        self.df = df
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.max_len = max_len
        self.width = width
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['dir_filelist']
        label = row['classID']
        mfcc = extract_mfcc(file_path, self.sr, self.n_mfcc, self.n_fft, self.n_hop)

        if mfcc is not None:
            # Pad or truncate the MFCC to ensure consistent length
            if mfcc.shape[1] < self.width:
                pad_width = self.width - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            elif mfcc.shape[1] > self.width:
                mfcc = mfcc[:, :self.width]

            mfcc = mfcc[np.newaxis, ...]  # Add a channel dimension
            if self.transform:
                mfcc = self.transform(mfcc)
            return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            # Handle the case where MFCC extraction fails
            return torch.zeros((1, self.n_mfcc, self.width)), torch.tensor(label, dtype=torch.long)

