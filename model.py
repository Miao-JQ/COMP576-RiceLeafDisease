from efficientnet_pytorch import EfficientNet
import torch

def initialize_model(efficient_name, num_classes=4, weights_path=None):
    """
    Initialize EfficientNet with optional pre-trained weights.
    
    Args:
        efficient_name (str): EfficientNet model name (e.g., 'efficientnet-b7').
        num_classes (int): Number of output classes.
        weights_path (str, optional): Path to pre-trained weights file. Defaults to None.
    
    Returns:
        EfficientNet: Initialized model.
    """
    # Initialize the model with default parameters
    model = EfficientNet.from_name(efficient_name)
    
    # Modify the classification head to match the number of classes
    model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
    
    # Load pre-trained weights if provided
    if weights_path:
        # Load state_dict from the pre-trained weights
        state_dict = torch.load(weights_path)
        
        # Remove '_fc' layer weights to avoid size mismatch
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_fc')}
        
        # Load weights into model
        model.load_state_dict(state_dict, strict=False)  # strict=False allows skipping mismatched keys
    
    return model
