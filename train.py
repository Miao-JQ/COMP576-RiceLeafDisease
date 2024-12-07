import os
from datetime import datetime
from utils import get_warmup_scheduler
import torch


def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_val_accuracy = 0.0

    # Prepare directories
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join(args.ckpt_dir, start_time)  # Save all checkpoints in a subdirectory with start time
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # Prepare log file
    log_file = os.path.join(args.logs_dir, f"training_log_{start_time}.txt")
    with open(log_file, "w") as log:
        log.write("Training Log\n")
        log.write(f"Start Time: {start_time}\n\n")

    # Warm-up setup
    scheduler = None
    if args.warm_up:
        warm_up_steps = len(train_loader) * 3  # 3 epochs for warm-up
        base_lr = optimizer.param_groups[0]['lr']
        scheduler = get_warmup_scheduler(optimizer, warm_up_steps, base_lr)

    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.warm_up and scheduler is not None:
                scheduler.step()  # Update learning rate during warm-up

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Print loss every 20 iterations
            if step % 20 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")
                with open(log_file, "a") as log:
                    log.write(f"Epoch [{epoch}/{args.epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}\n")

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, args.device)

        log_message = (
            f"Epoch [{epoch}/{args.epochs}]:\n"
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