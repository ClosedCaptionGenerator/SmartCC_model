import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from capsulelayers import CapsuleLayer
from model_train import train
from model_eval import evaluate
from hdf_train import get_train_loader
from hdf_validation import get_validation_loader
from config import NUM_CLASSES, TRAIN_PATH, VALID_PATH


# Model definition
class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes, in_channels, dropout_p):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsLayer(input_channels=256, dim_capsule=8, n_channels=32, kernel_size=9, stride=2, padding=0)
        self.digit_caps = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, routings=3, in_channels=32*6*6)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        x = self.digit_caps(x).view(x.size(0), -1)
        return self.decoder(x)


def main():
    # Initialize WandB
    wandb.init(project="train_caps_aws", entity="seoku", config={
        "out_channels": 16,  # Default value, will be overridden by sweep
        "dropout_p": 0.3,  # Default value, will be overridden by sweep
        "learning_rate": 1e-4,  # Default value, will be overridden by sweep
        "batch_size": 8  # Default value, will be overridden by sweep
    })
    config = wandb.config

    # Get hyperparameters from config
    num_routes = 2205
    in_channels = 100
    out_channels = config.out_channels
    dropout_p = config.dropout_p
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    # Get data loaders
    train_loader = get_train_loader(TRAIN_PATH, batch_size)
    val_loader = get_validation_loader(VALID_PATH, batch_size)

    # Initialize model, loss, optimizer, and scheduler
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CapsuleNetwork(NUM_CLASSES, num_routes, in_channels, out_channels, dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    wandb.watch(model, log="all")

    # Train and evaluate
    best_val_loss = float('inf')

    num_epochs = 50  # Use more epochs for the final training
    # grad_accum_steps = 4  # Set gradient accumulation steps
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, 1, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Log metrics to WandB
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            directory = "models"

            if not os.path.exists(directory):
                os.makedirs(directory)

            current = datetime.now().strftime('%Y%m%d_%H%M')

            file_path = os.path.join(directory, f'model_{current}.pth')
            torch.save(model.state_dict(), file_path)
            print(f"Best model saved with validation loss: {val_loss}")

    print('Finished Training')


if __name__ == '__main__':
    main()
