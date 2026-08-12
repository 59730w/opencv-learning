from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "assets"
OUTPUT_PATH = OUTPUT_DIR / "preprocessing_qa_20.png"

IMAGE_SIZE = (256, 256)
RANDOM_SEED = 42

# 与预训练 ResNet18 对应的 ImageNet 统计值
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_pair(image, trimap, do_flip):
    """对图像和mask执行完全相同的空间变换。"""

    # 图像允许双线性插值
    image = TF.resize(
        image,
        IMAGE_SIZE,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    # mask必须使用最近邻插值，防止产生新的标签值
    trimap = TF.resize(
        trimap,
        IMAGE_SIZE,
        interpolation=InterpolationMode.NEAREST,
    )

    # 同步水平翻转
    if do_flip:
        image = TF.hflip(image)
        trimap = TF.hflip(trimap)

    image = TF.pil_to_tensor(image).float() / 255.0

    # 原始trimap：
    # 1 = 宠物主体
    # 2 = 背景
    # 3 = 模糊边界
    raw_mask = torch.from_numpy(
        np.array(trimap, dtype=np.uint8, copy=True)
    )

    # 本项目的二值映射：
    # 背景2 → 0
    # 宠物主体1和模糊边界3 → 1
    binary_mask = (raw_mask != 2).float().unsqueeze(0)

    normalized_image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return normalized_image, raw_mask, binary_mask


def denormalize(image):
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1)


def main():
    dataset = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="segmentation",
        download=False,
    )

    rng = random.Random(RANDOM_SEED)
    indices = rng.sample(range(len(dataset)), 20)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(20, 4, figsize=(12, 60))

    all_raw_values = set()
    all_binary_values = set()

    for row, index in enumerate(indices):
        image, trimap = dataset[index]
        do_flip = rng.random() < 0.5

        image, raw_mask, binary_mask = preprocess_pair(
            image,
            trimap,
            do_flip,
        )

        all_raw_values.update(torch.unique(raw_mask).tolist())
        all_binary_values.update(torch.unique(binary_mask).tolist())

        display_image = denormalize(image).permute(1, 2, 0).numpy()
        binary_display = binary_mask.squeeze(0).numpy()

        axes[row, 0].imshow(display_image)
        axes[row, 0].set_title(f"Image #{index}, flip={do_flip}")

        axes[row, 1].imshow(raw_mask.numpy(), cmap="viridis", vmin=1, vmax=3)
        axes[row, 1].set_title("Raw trimap: 1 / 2 / 3")

        axes[row, 2].imshow(binary_display, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Binary mask: 0 / 1")

        axes[row, 3].imshow(display_image)
        axes[row, 3].imshow(
            binary_display,
            cmap="Reds",
            alpha=0.4,
            vmin=0,
            vmax=1,
        )
        axes[row, 3].set_title("Image + mask")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("samples checked:", len(indices))
    print("image shape:", tuple(image.shape))
    print("raw mask shape:", tuple(raw_mask.shape))
    print("binary mask shape:", tuple(binary_mask.shape))
    print("raw mask values:", sorted(all_raw_values))
    print("binary mask values:", sorted(all_binary_values))
    print("saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()