from dataset import get_data_loader


def get_train_loader(data_dir, batch_size, subset=None):
    return get_data_loader(data_dir, batch_size, shuffle=True, subset=subset)
