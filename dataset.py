import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class RiceDiseaseDataset(Dataset):
    def __init__(self, data_dir, target_shape=None):
        """
        Initializes the dataset by loading and processing images and labels.

        Args:
            data_dir (str): Path to the root directory containing subdirectories of images for each disease class.
            target_shape (tuple or None): Target shape for resizing images. If None, no resizing is performed.
        """
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.disease_dirs = {disease: os.path.join(data_dir, disease) for disease in os.listdir(data_dir)}
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self._load_data()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

    def _load_data(self):
        """
        Loads images and corresponding labels from the directories.
        """
        for label, directory in self.disease_dirs.items():
            if not os.path.isdir(directory):
                continue
            for file in os.listdir(directory):
                if file.endswith('.jpg') or file.endswith('.JPG'):
                    file_path = os.path.join(directory, file)
                    image = self._load_and_resize_image(file_path)
                    self.images.append(image)
                    self.labels.append(label)
        self.images = np.array(self.images)

    def _load_and_resize_image(self, file_path):
        """
        Loads an image and resizes it if target_shape is specified.

        Args:
            file_path (str): Path to the image file.

        Returns:
            torch.Tensor: Image tensor, optionally resized and permuted to CHW format.
        """
        image = cv2.imread(file_path)
        if image is not None:
            if self.target_shape is not None:
                image = cv2.resize(image, self.target_shape)  # Resize if target_shape is specified
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to CHW
            return image_tensor
        else:
            if self.target_shape is not None:
                return torch.zeros((3, self.target_shape[0], self.target_shape[1]), dtype=torch.float32)
            else:
                # If no target_shape is specified, return an empty image with the original shape
                return torch.zeros((3, image.shape[0], image.shape[1]), dtype=torch.float32)

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Image tensor.
            int: Encoded label.
        """
        image = self.images[idx]
        label = self.labels_encoded[idx]
        return image, label


# Example usage
if __name__ == "__main__":
    data_dir = './dataset'
    target_shape = None  # Set to None to avoid resizing

    # Initialize dataset
    dataset = RiceDiseaseDataset(data_dir, target_shape)

    # Split dataset
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

    sample_image, sample_label = train_dataset.__getitem__(0)
    print("Sample Image Size: ", sample_image.shape, "Label: ", sample_label)
