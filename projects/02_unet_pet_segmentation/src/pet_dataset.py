import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_SIZE = (256, 256)

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


class PetSegmentationDataset(Dataset):
    def __init__(self, root, split="trainval", training=False):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=False,
        )
        self.training = training

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, trimap = self.dataset[index]

        image = TF.resize(
            image,
            IMAGE_SIZE,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        # mask只能使用最近邻插值
        trimap = TF.resize(
            trimap,
            IMAGE_SIZE,
            interpolation=InterpolationMode.NEAREST,
        )

        # 图像和mask必须同步翻转
        if self.training and random.random() < 0.5:
            image = TF.hflip(image)
            trimap = TF.hflip(trimap)

        image = TF.pil_to_tensor(image).float() / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        raw_mask = torch.from_numpy(
            np.array(trimap, dtype=np.uint8, copy=True)
        )

        # 1=宠物，2=背景，3=模糊边界
        # 二值任务：1和3为前景，2为背景
        mask = (raw_mask != 2).float().unsqueeze(0)

        return image, mask