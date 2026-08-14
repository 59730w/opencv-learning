import torch

from src.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize_imagenet(image):
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=image.dtype,
        device=image.device,
    )[:, None, None]
    std = torch.tensor(
        IMAGENET_STD,
        dtype=image.dtype,
        device=image.device,
    )[:, None, None]
    return (image * std + mean).clamp(0.0, 1.0)
