import os

import cv2
import einops
import pydicom
from show import show

from dataset.crop_model_input import crop_diseased, crop_healthy
from dataset.csaw import CSAWDataset, get_aug
from dataset.dataloader import get_dataloader


def _image():
    return pydicom.dcmread(
        "/home/nikola/projects/pcm/data/csaw/images/00045_20990909_L_CC_1.dcm"
    ).pixel_array


def _mask():
    return cv2.imread(
        "/home/nikola/projects/pcm/data/csaw/masks/00045_20990909_L_CC_1_mask.png",
        cv2.IMREAD_GRAYSCALE,
    )


def _data_path():
    return "/home/nikola/projects/pcm/data/csaw/images"


def data():
    for entity in os.scandir(_data_path()):
        dicom_image = pydicom.dcmread(entity.path).pixel_array
        mask_path = entity.path.replace(".dcm", "_mask.png").replace("images", "masks")
        mask_image = cv2.imread(mask_path)
        show(dicom_image, mask_image)


def diseased_crop():
    image = _image()
    mask = _mask()
    for _ in range(5):
        patch, lc = crop_diseased(image, mask)
        show(patch, lc)


def healthy_crop():
    image = _image()
    for _ in range(5):
        patch, lc = crop_healthy(image)
        show(patch, lc)


def dataset():
    for entry in CSAWDataset(_data_path(), aug=get_aug(), diseased=True):
        x, y, mask = entry
        print(y.item())
        show(*x, mask)


def dataloader():
    for entry in get_dataloader(_data_path(), _data_path(), 1):
        x, y, mask = entry
        print(y.item())
        show(
            *einops.rearrange(x, "1 c h w -> c h w"),
            einops.rearrange(mask, "1 h w -> h w")
        )


if __name__ == "__main__":
    # data()
    # diseased_crop()
    # healthy_crop()
    dataset()
    # dataloader()
