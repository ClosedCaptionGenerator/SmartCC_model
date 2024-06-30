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

class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes, num_routes, in_channels, out_channels, dropout_p):
        super(CapsuleNetwork, self).__init__()
        self.capsule_layer = CapsuleLayer(num_capsules=num_classes, num_routes=num_routes, in_channels=in_channels, out_channels=out_channels)
        self.fc = nn.Linear(num_classes * num_routes * out_channels, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.capsule_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return x

def main(use_subset=False):
    # Initialize WandB
    wandb.init(project="optimize_project_aws2", entity="seoku", config={
        "out_channels": 32,  # Default value, will be overridden by sweep
        "dropout_p": 0.3,  # Default value, will be overridden by sweep
        "learning_rate": 1e-4,  # Default value, will be overridden by sweep
        "batch_size": 16  # Default value, will be overridden by sweep
    })
    config = wandb.config

    # Get hyperparameters from WandB config
    num_routes = 2205
    in_channels = 100
    out_channels = config.out_channels
    dropout_p = config.dropout_p
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    # Get data loaders
    subset = 0.1 if use_subset else None  # Use subset if specified
    train_loader = get_train_loader(TRAIN_PATH, batch_size, subset=subset)
    val_loader = get_validation_loader(VALID_PATH, batch_size, subset=subset)

    # Initialize model, loss, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CapsuleNetwork(NUM_CLASSES, num_routes, in_channels, out_channels, dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    wandb.watch(model, log="all")

    # Train and evaluate
    num_epochs = 10
    # best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, 1, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Log metrics to WandB
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        #         # Save the best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), f"best_model_{wandb.run.id}.pt")

    print('Finished Training')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_subset', action='store_true', help='Use a subset of the data for training')
    args = parser.parse_args()

    main(use_subset=args.use_subset)
