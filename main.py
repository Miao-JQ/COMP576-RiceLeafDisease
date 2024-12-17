from datasets import create_datasets, create_dataloader, load_data
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import FocalLoss, initialize_log_file, log_message, calculate_throughput, calculate_flops
from train import train
from test import test
from model import initialize_efficientnet, initialize_resnet50, initialize_vgg16
import json

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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.002)
    parser.add_argument('--warm_up', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--model_name', type=str, default='efficientnet')
    parser.add_argument('--efficient_name', type=str, default='efficientnet-b3')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--weight_dir', type=str, default='None')
    args = parser.parse_args()

    target_shape = efficientnet_input_sizes.get(args.efficient_name, (224, 224))
    images, labels = load_data(args.data_dir)

    train_data, val_data, test_data = create_datasets(images, labels)
    train_loader, val_loader, test_loader = create_dataloader(target_shape, train_data, val_data, test_data,
                                                              batch_size=args.batch_size)

    # Model setup
    if args.model_name == 'efficientnet':
        model = initialize_efficientnet(args.efficient_name, num_classes=4, weights_path=args.weight_dir)
    elif args.model_name == 'resnet50':
        model = initialize_resnet50()
    elif args.model_name == 'vgg16':
        model = initialize_vgg16()
    else:
        raise ValueError('Invalid model name')
    model.to(args.device)

    # Loss and optimizer
    criterion = FocalLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_file = initialize_log_file()
    # Log and print args
    args_dict = vars(args)
    args_message = "Training Configuration:\n" + json.dumps(args_dict, indent=4) + "\n"
    log_message(log_file, args_message)

    # Training loop
    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, log_file, args)

    # Test the model
    print("Evaluating on test dataset...")
    class_metrics, results = test(model, test_loader, log_file, args.device)
    print(f"Class metrics: {class_metrics}")
    print(f"Overall Accuracy: {results['overall']['accuracy']:.4f}")

    # Calculate throughput (images per second)
    print("Calculating throughput...")
    throughput = calculate_throughput(model, test_loader)
    throughput_message = f"Throughput: {throughput:.2f} images/sec\n"

    # Log throughput
    log_message(log_file, throughput_message)
