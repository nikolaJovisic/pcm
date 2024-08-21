import os
import pydicom

import matplotlib.pyplot as plt

data_dir = '/home/nikola/projects/pcm/data/csaw/images'

for entity in os.scandir(data_dir):
    dicom_image = pydicom.dcmread(entity.path)

    mask_path = entity.path.replace('.dcm', '_mask.png').replace('images', 'masks')
    
    mask_image = plt.imread(mask_path)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(dicom_image.pixel_array, cmap='gray')
    axs[0].set_title('DICOM Image')
    axs[0].axis('off')
    axs[1].imshow(mask_image, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    plt.show()
