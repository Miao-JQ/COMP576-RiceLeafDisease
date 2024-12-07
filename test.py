import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def test(model, test_loader, device, num_classes=4):
    model.eval()
    all_preds = []
    all_labels = []

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
        precision = precision_score(all_labels, all_preds, labels=[class_idx], average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, labels=[class_idx], average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, labels=[class_idx], average='binary', zero_division=0)
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

    return results