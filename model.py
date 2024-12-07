from efficientnet_pytorch import EfficientNet
import os
import torch
import torchvision.models as models
import torch.nn as nn


def initialize_efficientnet(efficient_name, num_classes=4, weights_path=None):
    if weights_path is not None:
        if os.path.isfile(weights_path):
            print(f"Loading weights from: {weights_path}")
            # Initialize model without downloading
            model = EfficientNet.from_name(efficient_name)
            model._fc = torch.nn.Linear(model._fc.in_features, num_classes)

            # Load state_dict from the provided weights file
            state_dict = torch.load(weights_path)
            # Remove '_fc' weights if they exist in the state_dict to prevent size mismatch
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_fc')}
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
    else:
        print(f"weights_path = {weights_path}")
        print(f"Downloading pre-trained weights for {efficient_name}...")
        # Automatically download pre-trained weights
        model = EfficientNet.from_pretrained(efficient_name, num_classes=num_classes)

    return model


def initialize_resnet50(num_classes=4):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Modify the classification head to match the number of output classes
    # ResNet-50's original fully connected layer is `model.fc`
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def initialize_vgg16(num_classes=4):
    # Load the pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Modify the classification head
    # The original VGG16 classifier is a Sequential block
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model