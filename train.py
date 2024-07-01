# main.py
import os
import torch
import argparse
import json
import wandb
from data_loader.dataset import prepare_data
from data_loader.data_loaders import create_dataloaders
from model.capsule_net import CapsuleNet
from trainers.trainer import train



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network")
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    input_shape = (config['n_mfcc'], 87)

    model = CapsuleNet(input_shape=input_shape, n_class=config['n_label'],routings=config['routings'])

    # train_data_paths = [os.path.join(config['train_data'], f) for f in os.listdir(config['train_data']) if f.endswith('.wav')]
    # val_data_paths = [os.path.join(config['val_data'], f) for f in os.listdir(config['val_data']) if f.endswith('.wav')]

    df_train = prepare_data(config['train_data'])
    df_val = prepare_data(config['val_data'])

    train_loader, val_loader = create_dataloaders(df_train, df_val, batch_size=config['batch_size'], sr=config['sample_rate'],
                                                  n_mfcc=config['n_mfcc'], n_fft=config['n_fft'], n_hop=config['n_hop'])

    wandb.init(project="aws-train-0701", config=config)
    trained_model = train(model, train_loader, val_loader, config)
    torch.save(trained_model.state_dict(), os.path.join(config['save_dir'], 'model.pth'))
