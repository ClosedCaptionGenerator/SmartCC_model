from dataset import get_data_loader


def get_validation_loader(data_dir, batch_size):
    return get_data_loader(data_dir, batch_size, shuffle=False)
