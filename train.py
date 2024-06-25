import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datetime import datetime
from capsulelayers import CapsuleLayer
from model_train import train
from model_eval import evaluate
from hdf_train import get_train_loader
from hdf_validation import get_validation_loader
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES, DROPOUT_RATE


print(f"MPS build check : {torch.backends.mps.is_built()}")
print(f"MPS available : {torch.backends.mps.is_available()}")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

wandb.init(
    project="capsuleNet",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "num_classes": NUM_CLASSES,
        "dropout_rate": DROPOUT_RATE,
    }
)


# Define the model architecture
class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.capsule_layer = CapsuleLayer(num_capsules=NUM_CLASSES, num_routes=2205, in_channels=100, out_channels=16)
        self.fc = nn.Linear(NUM_CLASSES * 2205 * 16, NUM_CLASSES)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.capsule_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return x


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device : {device}")

    train_loader = get_train_loader('data/train', BATCH_SIZE)
    val_loader = get_validation_loader('data/valid', BATCH_SIZE)

    model = CapsuleNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train(model, train_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss}')
        wandb.log({"validation_loss": val_loss, "epoch": epoch + 1})

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


if __name__ == '__main__':
    main()
