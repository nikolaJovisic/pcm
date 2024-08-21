import os

masks_path = "/home/nikola/projects/pcm/data/csaw/masks"
src_path = "/home/nikola/csaw/2021-204-1-1/data/"
dst_path = "/home/nikola/projects/pcm/data/csaw/images"

for entity in os.scandir(masks_path):
    image_name = entity.name[: -len("_mask.png")] + ".dcm"
    os.symlink(os.path.join(src_path, image_name), os.path.join(dst_path, image_name))
