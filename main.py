from dataset import RiceDiseaseDataset, create_dataset, create_dataloader
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from utils import FocalLoss, get_warmup_scheduler
from datetime import datetime


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, ckpt_dir, logs_dir, warm_up=False):
    """
    Train the model for a specified number of epochs and save checkpoints.
    Logs the training process and saves model checkpoints in a subdirectory with timestamps.
    """
    best_val_accuracy = 0.0

    # Prepare directories
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join(ckpt_dir, start_time)  # Save all checkpoints in a subdirectory with start time
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Prepare log file
    log_file = os.path.join(logs_dir, f"training_log_{start_time}.txt")
    with open(log_file, "w") as log:
        log.write("Training Log\n")
        log.write(f"Start Time: {start_time}\n\n")

    # Warm-up setup
    scheduler = None
    if warm_up:
        warm_up_steps = len(train_loader) * 3  # 3 epochs for warm-up
        base_lr = optimizer.param_groups[0]['lr']
        scheduler = get_warmup_scheduler(optimizer, warm_up_steps, base_lr)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if warm_up and scheduler is not None:
                scheduler.step()  # Update learning rate during warm-up

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Print loss every 50 iterations
            if step % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")
                with open(log_file, "a") as log:
                    log.write(f"Epoch [{epoch}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}\n")

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        log_message = (
            f"Epoch [{epoch}/{epochs}]:\n"
            f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n"
            f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n"
        )
        print(log_message)
        with open(log_file, "a") as log:
            log.write(log_message)

        # Save checkpoint with timestamp
        epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
        }, epoch_ckpt_path)
        print(f"Checkpoint saved: {epoch_ckpt_path}")
        with open(log_file, "a") as log:
            log.write(f"Checkpoint saved: {epoch_ckpt_path}\n")

        # Update best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print("Best model updated.")
            with open(log_file, "a") as log:
                log.write(f"Best model updated: {best_model_path}\n")


def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total
    return val_loss, val_accuracy


def test(model, test_loader, device):
    """
    Test the model on the test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='./dataset')
    arg_parser.add_argument('--batch_size', type=int, default=16)
    arg_parser.add_argument('--epochs', type=int, default=15)
    arg_parser.add_argument('--lr', type=float, default=0.005)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)
    arg_parser.add_argument('--warm_up', action='store_true')
    arg_parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
    arg_parser.add_argument('--logs_dir', type=str, default='./logs')
    arg_parser.add_argument('--efficient_name', type=str, default='efficientnet-b3')
    args = arg_parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    # Use recommended input size for EfficientNet
    efficientnet_input_sizes = {
        'efficientnet-b0': (224, 224),
        'efficientnet-b1': (240, 240),
        'efficientnet-b2': (260, 260),
        'efficientnet-b3': (300, 300),
        'efficientnet-b4': (380, 380),
        'efficientnet-b5': (456, 456),
        'efficientnet-b6': (528, 528),
        'efficientnet-b7': (600, 600),
    }
    target_shape = efficientnet_input_sizes.get(args.efficient_name, (224, 224))
    dataset = RiceDiseaseDataset(args.dataset, target_shape=target_shape)
    train_dataset, val_dataset, test_dataset = create_dataset(dataset)
    train_loader, val_loader, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset, args.batch_size)

    # Model setup
    model = EfficientNet.from_pretrained(args.efficient_name, num_classes=len(dataset.label_encoder.classes_))
    model = model.to(device)

    # Loss and optimizer
    criterion = FocalLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.ckpt_dir, args.logs_dir,
          warm_up=args.warm_up)

    # Test the model
    print("Evaluating on test dataset...")
    test_accuracy = test(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
