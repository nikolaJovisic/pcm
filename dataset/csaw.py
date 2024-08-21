import os

import cv2
import pydicom
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.crop_model_input import crop_diseased, crop_healthy
from dataset.preprocess import preprocess


class CSAWDataset(Dataset):
    def __init__(
        self,
        data_dir,
        healthy=True,
        model_dim=256,
        transform=transforms.Lambda(lambda x: x),
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.healthy = healthy
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
        image, _ = preprocess(image)
        image = self.transform(image)
        if self.healthy:
            patch, lc = crop_healthy(image, model_dim=self.model_dim)
        else:
            mask_path = image_path.replace(".dcm", "_mask.png").replace(
                "images", "masks"
            )
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, mask = preprocess(image, mask)
            mask = self.transform(mask)
            patch, lc = crop_diseased(image, mask, model_dim=self.model_dim)
        image = cv2.resize(image, (self.model_dim,) * 2)
        return patch, lc, image
