import os
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class S1OilDataset(Dataset):
    def __init__(self, data_dir, dataset_ids, transform=None, load_labels=True):
        self.image_dir = os.path.join(data_dir, 'image')
        self.mask_dir = os.path.join(data_dir, 'label')
        self.dataset_ids = dataset_ids
        self.transform = transform
        self.load_labels = load_labels

        self.image_files = []
        self.label_files = []

        for dataset_id in self.dataset_ids:
            image_file = os.path.join(self.image_dir, dataset_id)
            label_file = os.path.join(self.mask_dir, dataset_id)

            if os.path.exists(image_file):
                self.image_files.append(image_file)
                if self.load_labels and os.path.exists(label_file):
                    self.label_files.append(label_file)
                elif self.load_labels:
                    print(f"Warning: Label file not found: {label_file}")
            else:
                print(f"Warning: Image file not found: {image_file}")

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found for the given dataset IDs in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img = self.load_image(image_path)
        filename = os.path.basename(image_path)

        if not self.transform:
            return img, None, filename

        if self.load_labels: # supervised learning case
            label_path = self.label_files[idx]
            label = self.load_label(label_path)
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

            if isinstance(img, torch.Tensor):
                img = torch.clamp(img, 0, 1)
            else:
                img = np.clip(img, 0, 1)

            return img, label, filename

        else: # self-supervised learning case
            augmented1 = self.transform(image=img)
            augmented2 = self.transform(image=img)
            img1 = augmented1['image']
            img2 = augmented2['image']

            if isinstance(img1, torch.Tensor):
                img1 = torch.clamp(img1, 0, 1)
                img2 = torch.clamp(img2, 0, 1)
            else:
                img1 = np.clip(img1, 0, 1)
                img2 = np.clip(img2, 0, 1)

            return img1, img2, filename

    def load_image(self, path):
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)  # intensity
            img = img / 255.0  # normalize to [0,1]
        return img

    def load_label(self, path):
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.float32)
        return mask


def get_supervised_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])


def get_ssl_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),

        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

        A.ElasticTransform(alpha=100, sigma=100 * 0.05, alpha_affine=100 * 0.03, p=0.3),

        ToTensorV2()
    ])