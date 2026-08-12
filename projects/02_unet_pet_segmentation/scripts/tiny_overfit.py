from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pet_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PetSegmentationDataset,
)


DATA_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tiny_overfit"
ASSET_DIR = PROJECT_ROOT / "assets"

SEED = 42
TINY_SIZE = 8
BATCH_SIZE = 2
MAX_EPOCHS = 300
LEARNING_RATE = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits, masks, epsilon=1e-7):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + masks.sum(
        dim=(1, 2, 3)
    )

    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice.mean()


def calculate_metrics(model, loader, device):
    model.eval()

    intersection = 0
    prediction_sum = 0
    target_sum = 0
    union = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            predictions = (
                torch.sigmoid(model(images)) >= 0.5
            ).float()

            intersection += (predictions * masks).sum().item()
            prediction_sum += predictions.sum().item()
            target_sum += masks.sum().item()
            union += ((predictions + masks) > 0).sum().item()

    dice = (2 * intersection) / (
        prediction_sum + target_sum + 1e-7
    )
    iou = intersection / (union + 1e-7)

    return dice, iou


def save_predictions(model, loader, device, output_path):
    model.eval()

    images_list = []
    masks_list = []
    predictions_list = []

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            predictions = (
                torch.sigmoid(logits).cpu() >= 0.5
            ).float()

            images_list.append(images)
            masks_list.append(masks)
            predictions_list.append(predictions)

    images = torch.cat(images_list)
    masks = torch.cat(masks_list)
    predictions = torch.cat(predictions_list)

    fig, axes = plt.subplots(TINY_SIZE, 3, figsize=(9, 24))

    for row in range(TINY_SIZE):
        image = (
            images[row] * IMAGENET_STD + IMAGENET_MEAN
        ).clamp(0, 1)

        axes[row, 0].imshow(image.permute(1, 2, 0))
        axes[row, 0].set_title("Image")

        axes[row, 1].imshow(
            masks[row, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title("Ground truth")

        axes[row, 2].imshow(
            predictions[row, 0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title("Prediction")

        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    device = torch.device("cuda")

    # 不使用随机增广，确保每轮看到完全相同的8张图片
    full_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        training=False,
    )

    rng = random.Random(SEED)
    indices = rng.sample(range(len(full_dataset)), TINY_SIZE)
    tiny_dataset = Subset(full_dataset, indices)

    loader = DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    evaluation_loader = DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("tiny indices:", indices)
    print("tiny dataset size:", len(tiny_dataset))

    final_loss = None
    final_dice = 0
    final_iou = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0

        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = (
                bce_loss(logits, masks)
                + dice_loss(logits, masks)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        final_loss = total_loss / len(tiny_dataset)

        if epoch == 1 or epoch % 20 == 0:
            final_dice, final_iou = calculate_metrics(
                model,
                evaluation_loader,
                device,
            )

            print(
                f"epoch {epoch:03d} | "
                f"loss {final_loss:.4f} | "
                f"dice {final_dice:.4f} | "
                f"iou {final_iou:.4f}"
            )

            if final_dice >= 0.95 and final_iou >= 0.90:
                print("early stop: tiny-set target reached")
                break

    # 无论是否恰好落在20的倍数，都计算最终指标
    final_dice, final_iou = calculate_metrics(
        model,
        evaluation_loader,
        device,
    )

    checkpoint_path = OUTPUT_DIR / "unet_tiny_overfit.pth"
    prediction_path = ASSET_DIR / "tiny_overfit_predictions.png"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "indices": indices,
            "final_loss": final_loss,
            "final_dice": final_dice,
            "final_iou": final_iou,
        },
        checkpoint_path,
    )

    save_predictions(
        model,
        evaluation_loader,
        device,
        prediction_path,
    )

    print("final loss:", final_loss)
    print("final dice:", final_dice)
    print("final iou:", final_iou)
    print("checkpoint:", checkpoint_path)
    print("predictions:", prediction_path)

    if final_dice < 0.95 or final_iou < 0.90:
        raise RuntimeError(
            "极小集过拟合未达到验收标准，请不要开始完整训练"
        )

    print("tiny overfit test: PASSED")


if __name__ == "__main__":
    main()