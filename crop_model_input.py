import numpy as np
import cv2
import einops
import pydicom
import matplotlib.pyplot as plt

def crop_model_input(image, mask, max_expand=0.5):
    mask_indices = np.argwhere(mask)
    (ymin, xmin), (ymax, xmax) = mask_indices.min(0), mask_indices.max(0) + 1

    h = ymax - ymin
    w = xmax - xmin

    delta_h = h * np.random.uniform(0, max_expand)
    delta_w = w * np.random.uniform(0, max_expand)

    ymin = max(0, ymin - delta_h)
    ymax = min(image.shape[0], ymax + delta_h)
    xmin = max(0, xmin - delta_w)
    xmax = min(image.shape[1], xmax + delta_w)

    patch = image[int(ymin):int(ymax), int(xmin):int(xmax)]

    lc_ymin = max(0, ymin - h)
    lc_ymax = min(image.shape[0], ymax + h)
    lc_xmin = max(0, xmin - w)
    lc_xmax = min(image.shape[1], xmax + w)

    lc_patch = image[int(lc_ymin):int(lc_ymax), int(lc_xmin):int(lc_xmax)]

    return patch, lc_patch

if __name__ == '__main__':
    mask = cv2.imread('/home/nikola/projects/pcm/data/csaw/masks/00045_20990909_L_CC_1_mask.png', cv2.IMREAD_GRAYSCALE)
    image = pydicom.dcmread('/home/nikola/projects/pcm/data/csaw/images/00045_20990909_L_CC_1.dcm').pixel_array

    patch, lc = crop_model_input(image, mask)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(patch, cmap='gray')
    axs[0].set_title('Patch')
    axs[0].axis('off')
    axs[1].imshow(lc, cmap='gray')
    axs[1].set_title('Local Context')
    axs[1].axis('off')
    plt.show()
