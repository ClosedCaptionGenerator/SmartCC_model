# trainers/evaluator.py

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from model.capsule_net import CapsuleNet, margin_loss
from data_loader.data_loaders import create_dataloaders, prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def evaluate(model, test_loader, config):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, reconstruction = model(data)
            test_loss += margin_loss(target, output).item() + config['lam-recon'] * F.mse_loss(reconstruction, data).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return test_loss, accuracy, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, labels, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, labels, labels)
    plt.figure(figsize=(16, 8))
    sns.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 14}, fmt='g', linewidths=.5)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=400)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Capsule Network Evaluation")
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)

    model = CapsuleNet(input_shape=(config['n_mfcc'], 87), n_class=config['n_label'], routings=3).to(device)
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'model.pth')))

    val_data_paths = [os.path.join(config['val_data'], f) for f in os.listdir(config['val_data']) if f.endswith('.wav')]
    df_val = prepare_data(val_data_paths)
    val_loader = create_dataloaders(df_val, df_val, batch_size=config['batch_size'], sr=config['sample_rate'],
                                     n_mfcc=config['n_mfcc'], n_fft=config['n_fft'], n_hop=config['n_hop'])[0]

    test_loss, accuracy, y_pred, y_true = evaluate(model, val_loader, config)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {accuracy:.4f}')

    y_true = np.argmax(np.array(y_true), axis=1)
    y_pred = np.array(y_pred).flatten()

    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    print(classification_report(y_true, y_pred, target_names=labels))
    plot_confusion_matrix(y_true, y_pred, labels, config['save_dir'])

if __name__ == "__main__":
    main()
