# main.py
import os
import torch
import argparse
import json
import wandb
from torch.utils.data import DataLoader
from data_loader.dataset import AudioDataset, get_file_paths_and_labels
from model.capsulenet import CapsNet
from trainers.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network")
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device : {device}")

    # Update paths for train and validation data
    train_file_paths, train_labels = get_file_paths_and_labels(config['train_data'], config['label_mapping'])
    val_file_paths, val_labels = get_file_paths_and_labels(config['val_data'], config['label_mapping'])

    print(f"Number of train files: {len(train_file_paths)}")
    print(f"Number of validation files: {len(val_file_paths)}")

    # Define AudioDataset for train and validation data
    train_dataset = AudioDataset(train_file_paths, train_labels, sr=config['sample_rate'], n_mfcc=config['n_mfcc'],
                                 n_fft=config['n_fft'], n_hop=config['n_hop'], max_len=config['max_len'])
    val_dataset = AudioDataset(val_file_paths, val_labels, sr=config['sample_rate'], n_mfcc=config['n_mfcc'],
                               n_fft=config['n_fft'], n_hop=config['n_hop'], max_len=config['max_len'])

    # Define DataLoaders
    batch_size = config['batch_size']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Print dataset sample for debugging
    for mfcc, label in train_loader:
        print("Sample MFCC shape:", mfcc.shape)
        print("Sample label shape:", label.shape)
        break

    model = CapsNet(
        input_shape= mfcc.shape,
        n_class=config['n_label'],
        config=config).to(device)

    wandb.init(project="aws-train-0703", config=config)
    trained_model = train(model, train_loader, val_loader, config)
    torch.save(trained_model.state_dict(), os.path.join(config['save_dir'], 'model.pth'))
