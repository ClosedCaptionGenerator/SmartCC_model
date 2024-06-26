import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import wandb
from model.capsule_net import margin_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, config):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['lr-decay'])

    wandb.init(project="aws-train-0701", config=config)

    best_accuracy = 0.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0
        correct = 0

        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, reconstruction = model(data, target)

            loss = margin_loss(target, output) + config['lam-recon'] * nn.MSELoss()(reconstruction, data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()

        train_accuracy = correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader.dataset)

        print(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f} \tAccuracy: {train_accuracy:.4f}')

        val_loss, val_accuracy = validate(model, val_loader, config)
        print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                   "val_loss": val_loss, "val_accuracy": val_accuracy})

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

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, reconstruction = model(data)

            val_loss += margin_loss(target, output).item() + config['lam-recon'] * F.mse_loss(reconstruction, data).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    return val_loss, accuracy


# def train(model, train_loader, test_loader, config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     optimizer = optim.Adam(model.parameters(), lr=config['lr'])
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['lr-decay'])

#     for epoch in range(1, config['epochs'] + 1):
#         model.train()
#         total_loss = 0
#         correct = 0

#         for data, target in tqdm(train_loader):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output, reconstruction = model(data, target)

#             loss = margin_loss(target, output) + config['lam-recon'] * nn.MSELoss()(reconstruction, data)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()

#         print(f'Train Epoch: {epoch} \tLoss: {total_loss / len(train_loader.dataset):.6f}')

#         model.eval()
#         test_loss = 0
#         correct = 0

#         with torch.no_grad():
#             for data, target in test_loader:
#                 data, target = data.to(device), target.to(device)
#                 output, reconstruction = model(data, target)

#                 test_loss += margin_loss(target, output).item() + config['lam-recon'] * nn.MSELoss()(reconstruction, data).item()
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()

#         print(f'\nTest set: Average loss: {test_loss / len(test_loader.dataset):.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

#         scheduler.step()

#     return model
