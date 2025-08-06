import torch
import torchvision.transforms as T
from torchvision.transforms import Compose
import cv2
import numpy as np

def load_midas_model(device='cuda'):
    model_type = "DPT_Large"  # Can also be DPT_Hybrid or MiDaS_small
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform
