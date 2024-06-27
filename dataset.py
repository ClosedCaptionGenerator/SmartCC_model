import os
import json
import torch
import librosa
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, fixed_length=220500):
        self.root_dir = root_dir
        self.transform = transform
        self.fixed_length = fixed_length
        self.audio_files = []
        self.labels = []
        self.label_dict = self._create_label_dict()
        self._load_data()

    def _create_label_dict(self):
        return {
            '차량경적': 0,
            '차량사이렌': 1,
            '차량주행음': 2,
            '이륜차경적': 3,
            '이륜차주행음': 4,
            '비행기': 5,
            '헬리콥터': 6,
            '기차': 7,
            '지하철': 8,
            '발소리': 9,
            '가구소리': 10,
            '청소기': 11,
            '세탁기': 12,
            '개': 13,
            '고양이': 14,
            '공구': 15,
            '악기': 16,
            '항타기': 17,
            '파쇄기': 18,
            '콘크리트펌프': 19,
            '발전기': 20,
            '절삭기': 21,
            '송풍기': 22,
            '압축기': 23,
        }
    def _load_data(self):
        for label_type in ['교통소음', '사업장소음', '생활소음']:
            label_path = Path(self.root_dir) / 'label' / label_type
            raw_path = Path(self.root_dir) / 'raw' / label_type
            print(f'Scanning {label_type} and {raw_path}')
            for json_file in label_path.rglob('*.json'):
                with open(json_file) as f:
                    label_data = json.load(f)
                    annotations = label_data.get('annotations', [])
                    for annotation in annotations:
                        audio_file = json_file.stem + '_1.wav'
                        audio_file_path = raw_path / json_file.relative_to(label_path).parent / audio_file
                        if audio_file_path.exists():
                            self.audio_files.append(str(audio_file_path))
                            self.labels.append(annotation['categories']['category_03'])
                        else:
                            print(f"Audio file not found: {audio_file_path}")

            print(f"Loaded {len(self.audio_files)} audio files and {len(self.labels)} labels.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        audio, sr = librosa.load(audio_path, sr=None)
        audio = self._pad_or_truncate(audio)

        if self.transform:
            audio = self.transform(audio)

        audio = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(self._label_to_int(label), dtype=torch.long)

        return audio, label

    def _pad_or_truncate(self, audio):
        if len(audio) > self.fixed_length:
            # print(f"Truncating audio from {len(audio)} to {self.fixed_length}")
            return audio[:self.fixed_length]
        elif len(audio) < self.fixed_length:
            # print(f"Padding audio from {len(audio)} to {self.fixed_length}")
            return np.pad(audio, (0, self.fixed_length - len(audio)), mode='constant')
        return audio
    def _label_to_int(self, label):
        if label not in self.label_dict.keys():
            print(f'Unknown label: {label}')
        return self.label_dict.get(label, -1)


def get_data_loader(root_dir, batch_size, transform=None, shuffle=True, subset=None):
    dataset = AudioDataset(root_dir, transform)
    if subset:
        subset_size = int(len(dataset) * subset)
        dataset = Subset(dataset, range(subset_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
