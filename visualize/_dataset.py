import cv2
import pydicom
import matplotlib.pyplot as plt
from dataset.csaw import CSAWDataset
from dataset.crop_model_input import crop_healthy, crop_diseased
from show import show

def demo_diseased_crop(image, mask):
    patch, lc = crop_diseased(image, mask)
    show(patch, lc)

def demo_healthy_crop(image):
    patch, lc = crop_healthy(image)
    show(patch, lc)

def demo_dataset():
    dataset = CSAWDataset('/home/nikola/projects/pcm/data/csaw/images', healthy=True)

    for entry in dataset:
        show(*entry)

if __name__ == '__main__':
    demo_dataset()

    mask = cv2.imread('/home/nikola/projects/pcm/data/csaw/masks/00045_20990909_L_CC_1_mask.png', cv2.IMREAD_GRAYSCALE)
    image = pydicom.dcmread('/home/nikola/projects/pcm/data/csaw/images/00045_20990909_L_CC_1.dcm').pixel_array

    for i in range(5):
        demo_diseased_crop(image, mask)
    for i in range(5):
        demo_healthy_crop(image)
