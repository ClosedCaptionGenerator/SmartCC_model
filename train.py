# main.py
import os
import torch
import argparse
import json
import wandb
from data_loader.dataset import prepare_data
from data_loader.data_loaders import create_dataloaders
from model.capsulenet import CapsNet
from trainers.trainer import train


def get_wav_files(root_dir):
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network")
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_paths = get_wav_files(config['train_data'])
    val_data_paths = get_wav_files(config['val_data'])

    df_train = prepare_data(train_data_paths)
    df_val = prepare_data(val_data_paths)

    train_loader, val_loader, input_shape, n_class = create_dataloaders(df_train, df_val, batch_size=config['batch_size'], sr=config['sample_rate'],
                                                  n_mfcc=config['n_mfcc'], n_fft=config['n_fft'], n_hop=config['n_hop'], max_len=config['max_len'], width=config['width'])


    model = CapsNet(
        input_shape=input_shape,
        n_class=n_class,
        config=config).to(device)

    wandb.init(project="aws-train-0702", config=config)
    trained_model = train(model, train_loader, val_loader, config)
    torch.save(trained_model.state_dict(), os.path.join(config['save_dir'], 'model.pth'))
