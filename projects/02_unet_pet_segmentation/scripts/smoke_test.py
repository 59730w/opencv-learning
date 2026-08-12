from pathlib import Path
import random
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pet_dataset import PetSegmentationDataset


DATA_ROOT = PROJECT_ROOT / "datasets"
SEED = 42
BATCH_SIZE = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("本次冒烟测试要求CUDA可用")

    device = torch.device("cuda")

    dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        training=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    images, masks = next(iter(loader))

    print("dataset size:", len(dataset))
    print("image batch:", tuple(images.shape), images.dtype)
    print("mask batch:", tuple(masks.shape), masks.dtype)
    print("mask values:", torch.unique(masks).tolist())

    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)

    # 冒烟测试不下载预训练权重，只验证计算链路
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    tracked_parameter = next(model.parameters())
    parameter_before = tracked_parameter.detach().clone()

    logits = model(images)
    loss = loss_fn(logits, masks)

    if logits.shape != masks.shape:
        raise RuntimeError(
            f"输出和标签形状不一致：{logits.shape} != {masks.shape}"
        )

    if not torch.isfinite(loss):
        raise RuntimeError(f"loss不是有限数值：{loss.item()}")

    loss.backward()

    gradient_norm = tracked_parameter.grad.norm().item()
    optimizer.step()

    parameter_changed = not torch.equal(
        parameter_before,
        tracked_parameter.detach(),
    )

    print("device:", logits.device)
    print("logits shape:", tuple(logits.shape))
    print("loss:", loss.item())
    print("gradient norm:", gradient_norm)
    print("parameter changed:", parameter_changed)
    print("smoke test: PASSED")


if __name__ == "__main__":
    main()