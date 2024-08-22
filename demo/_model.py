import torch
from icecream import ic

from model.unet import Unet


def initialized_inference():
    model = Unet(dim=128, dim_mults=(1, 2, 2, 4, 4), channels=3)
    model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ic(model(x))


def weight_loading():
    model = Unet(dim=128, dim_mults=(1, 2, 2, 4, 4), channels=3)
    model.eval()
    model_checkpoint = torch.load(f"model/model_131999.pt", map_location="cpu")
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
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ic(model(x))


if __name__ == "__main__":
    # initialized_inference()
    weight_loading()
