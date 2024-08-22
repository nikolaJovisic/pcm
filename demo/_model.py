from model.unet import Unet
import torch
from icecream import ic

def initialized_inference():
    model = Unet(dim=128, dim_mults=(1, 2, 2, 4, 4), channels=3)
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    ic(y.shape, y.min(), y.max())
    

if __name__ == '__main__':
    initialized_inference()