from dataset import RiceDiseaseDataset, create_dataset, create_dataloader
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from utils import FocalLoss


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, ckpt_dir):
    """
    Train the model for a specified number of epochs and save checkpoints.
    """
    best_val_accuracy = 0.0
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Validation phase
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

        print(f"Epoch [{epoch}/{epochs}]:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Update best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            print("Best model updated.")


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
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--epochs', type=int, default=10)
    arg_parser.add_argument('--lr', type=float, default=0.001)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)
    arg_parser.add_argument('--warm_up', action='store_true')
    arg_parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
    args = arg_parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    dataset = RiceDiseaseDataset(args.dataset, target_shape=(224, 224))
    train_dataset, val_dataset, test_dataset = create_dataset(dataset)
    train_loader, val_loader, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset, args.batch_size)

    # Model setup
    model = resnet18(pretrained=False, num_classes=len(dataset.label_encoder.classes_))
    model = model.to(device)

    # Loss and optimizer
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.ckpt_dir)

    # Test the model
    print("Evaluating on test dataset...")
    test_accuracy = test(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
