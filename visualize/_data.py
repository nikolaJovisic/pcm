import os

import matplotlib.pyplot as plt
import pydicom
from show import show

data_dir = "/home/nikola/projects/pcm/data/csaw/images"

for entity in os.scandir(data_dir):
    dicom_image = pydicom.dcmread(entity.path).pixel_array
    mask_path = entity.path.replace(".dcm", "_mask.png").replace("images", "masks")
    mask_image = plt.imread(mask_path)

    show(dicom_image, mask_image)
