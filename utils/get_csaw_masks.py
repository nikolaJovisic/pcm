import os
import cv2
import numpy as np

src_path = '/home/nikola/csaw/2021-204-1-1/data/anon_annotations_nonhidden'
dst_path = '/home/nikola/projects/pcm/data/csaw/masks'

for entity in os.scandir(src_path):
    if not entity.name.endswith('.png'):
        continue
    mask = cv2.imread(entity.path, cv2.IMREAD_GRAYSCALE)
    if np.sum(mask) != 255:
        os.symlink(entity.path, os.path.join(dst_path, entity.name))