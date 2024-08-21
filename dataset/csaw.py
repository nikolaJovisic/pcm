import os

import albumentations as A
import cv2
import einops
import numpy as np
import pydicom
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from dataset.crop_model_input import crop_diseased, crop_healthy
from dataset.preprocess import preprocess


def get_x_aug():
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 5), p=0.3),
        A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
        ToTensorV2(),
    ])

def uint16_to_float32_min_max(image):
    image = image.astype(np.float32)
    return (image - np.min(image)) / (np.max(image) - np.min(image))

class CSAWDataset(Dataset):
    def __init__(
        self,
        data_dir,
        diseased,
        model_dim=256,
        x_aug=A.Compose([ToTensorV2()]),
    ):
        self.data_dir = data_dir
        self.x_augmentations = x_aug
        self.diseased = diseased
        self.model_dim = model_dim
        self.images = [
            entity.path
            for entity in os.scandir(data_dir)
            if entity.name.endswith(".dcm")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = pydicom.dcmread(image_path).pixel_array
        image = uint16_to_float32_min_max(image)
        if self.diseased:
            mask_path = image_path.replace(".dcm", "_mask.png").replace(
                "images", "masks"
            )
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image, mask = preprocess(image, mask)
            patch, lc = crop_diseased(image, mask, model_dim=self.model_dim)
        else:
            image, mask = preprocess(image)
            patch, lc = crop_healthy(image, model_dim=self.model_dim)
        
        image = cv2.resize(image, (self.model_dim,) * 2)
        mask = cv2.resize(mask, (self.model_dim,) * 2)

        x = einops.rearrange([patch, lc, image], "c h w -> h w c")

        aug = self.x_augmentations(image=x, mask=mask)

        x = aug["image"]
        mask = aug["mask"]

        y = torch.tensor(int(self.diseased))

        return x, y, mask