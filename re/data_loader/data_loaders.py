from torch.utils.data import DataLoader
from .dataset import AudioDataset

def get_data_loader(root_dir, batch_size, transform=None, shuffle=True, subset=None):
    dataset = AudioDataset(root_dir, transform)
    if subset:
        subset_size = int(len(dataset) * subset)
        dataset = Subset(dataset, range(subset_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
