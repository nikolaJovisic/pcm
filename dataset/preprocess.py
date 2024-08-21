import cv2
import numpy as np
from skimage.filters import threshold_otsu


def add_one_if_even(number):
    if number % 2 == 0:
        return number + 1
    else:
        return number

def create_kernel(img, factor):
    return add_one_if_even(img.shape[0] // factor), add_one_if_even(img.shape[1] // factor)

def binarize(img):
    b_img = img.astype(np.float32)
    b_img = 255 * (b_img - np.min(b_img)) / (np.max(b_img) - np.min(b_img))
    b_img = b_img.astype(np.uint8)

    blured = cv2.GaussianBlur(b_img, create_kernel(b_img, 50), 0)

    otsu_tr = threshold_otsu(blured) * 0.175
    mask = np.where(blured >= otsu_tr, 1, 0).astype(np.uint8)

    return mask


def dilate(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, create_kernel(mask, 200))
    dilated_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return dilated_mask

def erode(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, create_kernel(mask, 500))
    eroded_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    return eroded_mask

def keep_largest_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_blob_mask = np.zeros_like(mask)
    cv2.drawContours(largest_blob_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    return largest_blob_mask


def get_breast_mask(image):
    mask = binarize(image)
    mask = erode(mask)
    mask = dilate(mask)
    return keep_largest_blob(mask)


def keep_only_breast(image):
    mask = get_breast_mask(image)
    return image * mask, mask


def negate_if_should(image):
    hist, bins = np.histogram(image.ravel(), bins=2, range=[image.min(), image.max()])
    return image if hist[0] > hist[-1] else np.max(image) - image

def crop_breast(image, breast_mask, mass_mask):
    breast_mask = get_breast_mask(image)
    breast_mask_indices = np.argwhere(breast_mask)

    (ymin, xmin), (ymax, xmax) = breast_mask_indices.min(0), breast_mask_indices.max(0) + 1

    ymin = max(0, ymin)
    ymax = min(image.shape[0], ymax)
    xmin = max(0, xmin)
    xmax = min(image.shape[1], xmax)

    return image[ymin:ymax, xmin:xmax], mass_mask[ymin:ymax, xmin:xmax]


def preprocess(image, mass_mask=None):
    if mass_mask == None:
        mass_mask = np.zeros_like(image)
    image = negate_if_should(image)
    image, breast_mask = keep_only_breast(image)
    return crop_breast(image, breast_mask, mass_mask)
