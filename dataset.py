import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

class RiceDiseaseDataset(Dataset):
    def __init__(self, data_dir, target_shape=None, augment=False):
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.augment = augment
        self.disease_dirs = {disease: os.path.join(data_dir, disease) for disease in os.listdir(data_dir)}
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self._load_data()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

        # Define transforms
        self.transform = self._get_transforms()

    def _load_data(self):
        for label, directory in self.disease_dirs.items():
            if not os.path.isdir(directory):
                continue
            for file in os.listdir(directory):
                if file.endswith('.jpg') or file.endswith('.JPG'):
                    file_path = os.path.join(directory, file)
                    self.images.append(file_path)
                    self.labels.append(label)

    def _get_transforms(self):
        if self.augment:
            return transforms.Compose([
                transforms.Resize(self.target_shape) if self.target_shape else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.target_shape) if self.target_shape else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels_encoded[idx]

        # Load image using PIL for compatibility with torchvision.transforms
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        return image, label



def create_dataset(dataset):
    train_size = int(0.64 * len(dataset))
    val_size = int(0.16 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    print("Number of All Images: ", len(dataset))
    print("Number of Training Images: ", len(train_dataset))
    print("Number of Validation Images: ", len(val_dataset))
    print("Number of Test Images: ", len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def create_dataloader(train_dataset, val_dataset, test_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader
