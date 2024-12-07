import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image


def load_data(data_dir):
    images = []
    labels = []
    disease_dirs = {disease: os.path.join(data_dir, disease) for disease in os.listdir(data_dir)}

    for label, directory in disease_dirs.items():
        if not os.path.isdir(directory):
            continue
        for file in os.listdir(directory):
            if file.endswith('.jpg') or file.endswith('.JPG'):
                file_path = os.path.join(directory, file)
                images.append(file_path)
                labels.append(label)

    return images, labels


def create_datasets(images, labels, train_ratio=0.64, val_ratio=0.16):
    dataset_size = len(images)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = ([images[i] for i in train_indices], [labels[i] for i in train_indices])
    val_data = ([images[i] for i in val_indices], [labels[i] for i in val_indices])
    test_data = ([images[i] for i in test_indices], [labels[i] for i in test_indices])

    return train_data, val_data, test_data


class RiceDiseaseDataset(Dataset):
    def __init__(self, dataset, target_shape=None, augment=False):
        """
        Dataset for rice disease classification.

        Args:
            dataset (tuple): A tuple containing (images, labels) loaded externally.
            target_shape (tuple, optional): Desired image shape (height, width). Defaults to None.
            augment (bool): Whether to apply data augmentation. Defaults to False.
        """
        self.images, self.labels = dataset
        self.target_shape = target_shape
        self.augment = augment
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

        # Define transforms
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """
        Define the transformations for preprocessing and augmentation.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        if self.augment:
            return transforms.Compose([
                transforms.Resize(self.target_shape) if self.target_shape else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4444, 0.5317, 0.3068], std=[0.1910, 0.1866, 0.1844])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.target_shape) if self.target_shape else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4444, 0.5317, 0.3068], std=[0.1910, 0.1866, 0.1844])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            torch.Tensor: Transformed image tensor.
            int: Encoded label corresponding to the image.
        """
        image_path = self.images[idx]
        label = self.labels_encoded[idx]

        # Load image using PIL for compatibility with torchvision.transforms
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        return image, label


def create_dataloader(train_data, val_data, test_data, batch_size, num_workers=0):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        train_data (tuple): Training dataset as (images, labels).
        val_data (tuple): Validation dataset as (images, labels).
        test_data (tuple): Test dataset as (images, labels).
        batch_size (int): Batch size for the DataLoader.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 0.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
    train_dataset = RiceDiseaseDataset(train_data, target_shape=(224, 224), augment=True)
    val_dataset = RiceDiseaseDataset(val_data, target_shape=(224, 224), augment=False)
    test_dataset = RiceDiseaseDataset(test_data, target_shape=(224, 224), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
