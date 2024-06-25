import torch
import wandb

def train(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                avg_loss = running_loss / 100
                print(f"[Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}")
                wandb.log({"batch_loss": avg_loss})
                running_loss = 0.0

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

        scheduler.step()

    print('Finished Training')
