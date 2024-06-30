import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from data_loader.dataset import get_data_loader
from models.capsule_net import capsule_net

class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = capsule_net(num_classes=24).to(device)  # Ensure correct number of classes
        self.criterion = nn.CrossEntropyLoss()  # Adjust based on specific loss requirement
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

        self.train_loader = get_data_loader(config['train_data'], config['batch_size'])
        self.val_loader = get_data_loader(config['val_data'], config['batch_size'], shuffle=False)

        # Initialize wandb
        wandb.init(project='capsule_network_project', config=config)

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output, reconstructions = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss = self.validate()

            # Log metrics to wandb
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

            print(f'Epoch [{epoch+1}/{self.config["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

        # Finish wandb session
        wandb.finish()

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, reconstructions = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)
