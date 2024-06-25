import torch
import torch.nn as nn
import torch.optim as optim
import time
from capsulelayers import CapsuleLayer
from hdf_train import get_train_loader
from config import BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, NUM_EPOCHS


print(f"MPS build check : {torch.backends.mps.is_built()}")
print(f"MPS available : {torch.backends.mps.is_available()}")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Define the model architecture
class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.capsule_layer = CapsuleLayer(num_capsules=NUM_CLASSES, num_routes=2205, in_channels=100, out_channels=16)
        self.fc = nn.Linear(NUM_CLASSES * 2205 * 16, NUM_CLASSES)

    def forward(self, x):
        x = self.capsule_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_time = time.time() - start_time
    return running_loss / len(dataloader), epoch_time


def profile_training():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device : {device}")

    # Load a small subset of the data
    train_loader = get_train_loader('data/train', BATCH_SIZE)
    subset_size = int(0.001 * len(train_loader.dataset))  # 0.1% of the dataset
    subset_indices = list(range(subset_size))
    subset_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_loader.dataset, subset_indices),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = CapsuleNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Profile training for a few epochs
    num_epochs = 3
    total_time = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        avg_loss, epoch_time = train_one_epoch(model, subset_loader, criterion, optimizer, device)
        total_time += epoch_time
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    # Estimate total training time for the full dataset
    estimated_time_per_epoch = (total_time / num_epochs) / 0.001
    total_estimated_time = estimated_time_per_epoch * NUM_EPOCHS
    print(f"Estimated total training time for full dataset: {total_estimated_time / 3600:.2f} hours for {NUM_EPOCHS} epochs")


if __name__ == '__main__':
    profile_training()
