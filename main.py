from dataset import RiceDiseaseDataset, create_datasets, create_dataloader, load_data
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import FocalLoss
from train import train
from test import test
from model import initialize_model
from utils import calculate_mean_std

# Dataset and DataLoader
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--warm_up', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--efficient_name', type=str, default='efficientnet-b3')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_shape = efficientnet_input_sizes.get(args.efficient_name, (224, 224))
    images, labels = load_data(args.data_dir)

    train_data, val_data, test_data = create_datasets(images, labels)
    train_loader, val_loader, test_loader = create_dataloader(train_data, val_data, test_data, batch_size=16)

    # Model setup
    model = initialize_model(args.efficient_name, num_classes=4)
    model.to(args.device)

    # Loss and optimizer
    criterion = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, args)

    # Test the model
    print("Evaluating on test dataset...")
    results = test(model, test_loader, args.device)
    print(f"Overall Accuracy: {results['overall']['accuracy']:.4f}")
