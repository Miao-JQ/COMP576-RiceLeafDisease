from efficientnet_pytorch import EfficientNet


def initialize_model(efficient_name, num_classes=4):
    model = EfficientNet.from_pretrained(efficient_name, num_classes=num_classes)
    return model
