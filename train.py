import os
from datetime import datetime
import torch
from torch.optim import lr_scheduler
from utils import get_warmup_scheduler, log_message


def train(model, train_loader, val_loader, criterion, optimizer, log_file, args):
    best_val_accuracy = 0.0

    # Learning rate scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6, 8], gamma=0.3)

    # Prepare directories for saving checkpoints and logs
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join(args.ckpt_dir, start_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # Warm-up scheduler setup
    warmup_scheduler = None
    if args.warm_up:
        warm_up_steps = len(train_loader) * 3  # Warm-up for 3 epochs
        base_lr = optimizer.param_groups[0]['lr']
        warmup_scheduler = get_warmup_scheduler(optimizer, warm_up_steps, base_lr)

    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            # Move data to device
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)

            # Compute loss and backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update learning rate scheduler
            scheduler.step()
            if args.warm_up and warmup_scheduler is not None:
                warmup_scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log loss every 20 iterations
            if step % 20 == 0:
                msg = (f"Epoch [{epoch}/{args.epochs}], Step [{step}/{len(train_loader)}], "
                       f"Loss: {loss.item():.4f}")
                log_message(log_file, msg)

        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, args.device)

        # Log epoch results
        log_message(log_file, (
            f"Epoch [{epoch}/{args.epochs}]:\n"
            f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n"
            f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n"
        ))

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
        log_message(log_file, f"Checkpoint saved: {epoch_ckpt_path}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            log_message(log_file, f"Best model updated: {best_model_path}")



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