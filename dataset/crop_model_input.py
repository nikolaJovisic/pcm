import cv2
import numpy as np


def crop_diseased(image, mask, max_expand=0.2, model_dim=256):
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

    patch = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]

    lc_ymin = max(0, ymin - h)
    lc_ymax = min(image.shape[0], ymax + h)
    lc_xmin = max(0, xmin - w)
    lc_xmax = min(image.shape[1], xmax + w)

    lc = image[int(lc_ymin) : int(lc_ymax), int(lc_xmin) : int(lc_xmax)]

    patch = cv2.resize(patch, (model_dim,) * 2)
    lc = cv2.resize(lc, (model_dim,) * 2)

    return patch, lc


def crop_healthy(image, max_expand=0.5, min_patch_len=196, epsilon=0.3, model_dim=256):
    patch_len = int(min_patch_len * np.random.uniform(1, 1 + max_expand))

    while True:
        patch, lc = _try_crop_healthy(image, patch_len)
        if np.mean(patch) > epsilon * np.max(image):
            patch = cv2.resize(patch, (model_dim,) * 2)
            lc = cv2.resize(lc, (model_dim,) * 2)
            return patch, lc


def _try_crop_healthy(image, patch_len):
    x = np.random.randint(0, image.shape[1] - patch_len)
    y = np.random.randint(0, image.shape[0] - patch_len)

    patch = image[y : y + patch_len, x : x + patch_len]

    lc_ymin = max(0, y - patch_len)
    lc_ymax = min(image.shape[0], y + 2 * patch_len)
    lc_xmin = max(0, x - patch_len)
    lc_xmax = min(image.shape[1], x + 2 * patch_len)

    lc = image[lc_ymin:lc_ymax, lc_xmin:lc_xmax]

    return patch, lc
