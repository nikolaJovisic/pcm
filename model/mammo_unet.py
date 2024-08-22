import torch

from model.unet import Unet
from settings import settings


def get_mammo_unet():
    model = Unet(dim=128, dim_mults=(1, 2, 2, 4, 4), channels=3)
    model_checkpoint = torch.load(settings.mammo_unet_weights_path, map_location="cpu")
    model.load_state_dict(
        (
            dict(
                [
                    (n.replace("module.", ""), p)
                    for n, p in model_checkpoint["model_state"].items()
                ]
            )
        )
    )
    return model
