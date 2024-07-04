import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm
import wandb
from utils.utils import CapsuleLoss

scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, config):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['lr-decay'])
    loss_fn = CapsuleLoss(lam_recon=config['lam_recon'])

    wandb.init(project="aws-train-0704", config=config)
    best_accuracy = 0.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0
        correct = 0

        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            target_one_hot = F.one_hot(target, num_classes=config['n_label']).float()
            optimizer.zero_grad()

            # amp
            with autocast():
                output, reconstructions = model(data)
                loss = loss_fn(target_one_hot, output, reconstructions, data)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            lengths = torch.sqrt((output ** 2).sum(dim=2))
            _, pred = lengths.max(dim=1)
            # _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()


        train_accuracy = correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader.dataset)
        print(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f} \tAccuracy: {train_accuracy:.4f}')

        val_loss, val_accuracy = validate(model, val_loader, config)
        print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})

        if val_accuracy > best_accuracy:
            t = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(config['save_dir'], f'best_model_{t}.pth'))
            print(f'Saved model with accuracy: {best_accuracy:.4f} at best_model_{t}')

        scheduler.step()

    return model


def validate(model, val_loader, config):
    model.eval()
    val_loss = 0
    correct = 0
    loss_fn = CapsuleLoss(lam_recon=config['lam_recon'])


    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target_one_hot = F.one_hot(target, num_classes=config['n_label']).float()
            output, reconstructions = model(data)
            val_loss += loss_fn(target_one_hot, output, reconstructions, data)
            lengths = torch.sqrt((output ** 2).sum(dim=2))
            _, predicted = lengths.max(dim=1)
            correct += (predicted == target).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    return val_loss, accuracy

