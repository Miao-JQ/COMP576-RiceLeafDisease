import argparse
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from model import initialize_resnet50, initialize_efficientnet, initialize_vgg16


def load_image(image_path, target_shape=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4444, 0.5317, 0.3068], std=[0.1910, 0.1866, 0.1844])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def load_model(model_name, ckpt_path, num_classes=4, device='cuda', efficient_name='efficientnet-b3'):
    # Select model based on model_name
    if model_name == 'resnet50':
        model = initialize_resnet50(num_classes=num_classes)
    elif model_name == 'efficientnet':
        model = initialize_efficientnet(efficient_name, num_classes=num_classes)
    elif model_name == 'vgg16':
        model = initialize_vgg16(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Load the checkpoint weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def classify_image(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)  # Convert logits to probabilities
    return probabilities.cpu().numpy()


def save_results(probabilities, output_path, class_names):
    with open(output_path, "w") as f:
        print("Prediction Probabilities:")
        f.write("Prediction Probabilities:\n")
        for i, prob in enumerate(probabilities):
            print(f"{class_names[i]}: {prob:.4f}")
            f.write(f"{class_names[i]}: {prob:.4f}\n")
    print(f"Results saved to {output_path}")


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

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]  # Update with actual class names

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to the image")
    parser.add_argument("--ckpt_path", required=True, type=str, help="Path to the checkpoint")
    parser.add_argument("--output_path", type=str, default="./classify_result.txt", help="Path to the output file")
    parser.add_argument("--model_name", type=str, default='efficientnet',
                        choices=['resnet50', 'efficientnet', 'vgg16'], help="Model to use for inference")
    parser.add_argument("--efficient_name", type=str, default='efficientnet-b3', help="EfficientNet version")
    args = parser.parse_args()

    # Determine target shape for EfficientNet
    target_shape = efficientnet_input_sizes.get(args.efficient_name, (224, 224))

    # Load and preprocess the image
    print("Loading image...")
    image_tensor = load_image(args.input_path, target_shape=target_shape)

    # Load the model
    print(f"Loading {args.model_name} model...")
    model = load_model(args.model_name, args.ckpt_path, num_classes=len(CLASS_NAMES),
                       device=DEVICE, efficient_name=args.efficient_name)

    # Perform classification
    print("Classifying image...")
    probabilities = classify_image(image_tensor, model, DEVICE)

    # Save results
    save_results(probabilities, args.output_path, CLASS_NAMES)
