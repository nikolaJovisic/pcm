import torch
from icecream import ic

from model.mammo_unet import get_mammo_unet
from model.unet import Unet


def uninitialized_inference():
    model = Unet(dim=128, dim_mults=(1, 2, 2, 4, 4), channels=3)
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ic(model(x))


def weights_loaded_inference():
    model = get_mammo_unet()
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ic(model(x))


if __name__ == "__main__":
    # uninitialized_inference()
    weights_loaded_inference()
