import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import log_message


def test(model, test_loader, log_file, device, num_classes=4):
    """
    Test the model and calculate evaluation metrics.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        log_file: Path to the log file for recording results.
        device: Device to run the evaluation on (CPU or GPU).
        num_classes: Number of classes in the dataset.

    Returns:
        dict: Per-class metrics.
        dict: Overall metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    log_message(log_file, "Starting evaluation on test dataset...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Calculate per-class metrics
    class_metrics = {}
    for class_idx in range(num_classes):
        precision = precision_score(
            all_labels, all_preds, labels=[class_idx], average=None, zero_division=0
        )[0]
        recall = recall_score(
            all_labels, all_preds, labels=[class_idx], average=None, zero_division=0
        )[0]
        f1 = f1_score(
            all_labels, all_preds, labels=[class_idx], average=None, zero_division=0
        )[0]
        class_metrics[class_idx] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    # Combine results into a dictionary
    results = {
        'overall': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
        },
        'per_class': class_metrics
    }

    # Log overall metrics
    log_message(log_file, "Overall Test Results:")
    log_message(log_file, f"  Accuracy: {overall_accuracy:.4f}")
    log_message(log_file, f"  Precision: {overall_precision:.4f}")
    log_message(log_file, f"  Recall: {overall_recall:.4f}")
    log_message(log_file, f"  F1 Score: {overall_f1:.4f}")

    # Log per-class metrics
    log_message(log_file, "Per-Class Test Results:")
    for class_idx, metrics in class_metrics.items():
        log_message(
            log_file,
            f"  Class {class_idx} -> Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
        )

    return class_metrics, results
