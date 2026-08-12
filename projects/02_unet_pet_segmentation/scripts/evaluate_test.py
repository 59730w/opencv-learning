from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pet_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PetSegmentationDataset,
)


DATA_ROOT = PROJECT_ROOT / "datasets"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "baseline" / "best_model.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_evaluation"
ASSET_PATH = PROJECT_ROOT / "assets" / "test_cases.png"

BATCH_SIZE = 4
THRESHOLD = 0.5
EPSILON = 1e-7


def create_model(device):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("checkpoint epoch:", checkpoint["epoch"])
    print("checkpoint val dice:", checkpoint["best_val_dice"])

    return model


def evaluate(model, loader, device):
    global_intersection = 0.0
    global_prediction_sum = 0.0
    global_target_sum = 0.0
    global_union = 0.0

    sample_records = []
    sample_index = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(images)

            predictions = (
                torch.sigmoid(logits) >= THRESHOLD
            ).float()

            batch_intersections = (
                predictions * masks
            ).sum(dim=(1, 2, 3))

            batch_prediction_sums = predictions.sum(
                dim=(1, 2, 3)
            )

            batch_target_sums = masks.sum(
                dim=(1, 2, 3)
            )

            batch_unions = (
                (predictions + masks) > 0
            ).sum(dim=(1, 2, 3))

            batch_dice = (
                2 * batch_intersections + EPSILON
            ) / (
                batch_prediction_sums
                + batch_target_sums
                + EPSILON
            )

            batch_iou = (
                batch_intersections + EPSILON
            ) / (
                batch_unions + EPSILON
            )

            global_intersection += batch_intersections.sum().item()
            global_prediction_sum += batch_prediction_sums.sum().item()
            global_target_sum += batch_target_sums.sum().item()
            global_union += batch_unions.sum().item()

            images_cpu = images.cpu()
            masks_cpu = masks.cpu()
            predictions_cpu = predictions.cpu()

            for position in range(images.size(0)):
                sample_records.append(
                    {
                        "index": sample_index,
                        "dice": float(batch_dice[position].item()),
                        "iou": float(batch_iou[position].item()),
                        "image": images_cpu[position],
                        "mask": masks_cpu[position],
                        "prediction": predictions_cpu[position],
                    }
                )
                sample_index += 1

    global_dice = (
        2 * global_intersection
    ) / (
        global_prediction_sum
        + global_target_sum
        + EPSILON
    )

    global_iou = global_intersection / (
        global_union + EPSILON
    )

    return global_dice, global_iou, sample_records


def select_cases(records):
    ordered = sorted(records, key=lambda item: item["dice"])

    worst = ordered[:3]
    best = ordered[-3:][::-1]

    middle = len(ordered) // 2
    median = ordered[middle - 1:middle + 2]

    return [
        ("Worst", record) for record in worst
    ] + [
        ("Median", record) for record in median
    ] + [
        ("Best", record) for record in best
    ]


def save_cases(cases, output_path):
    fig, axes = plt.subplots(
        len(cases),
        4,
        figsize=(12, 3 * len(cases)),
    )

    for row, (group, record) in enumerate(cases):
        image = (
            record["image"] * IMAGENET_STD
            + IMAGENET_MEAN
        ).clamp(0, 1)

        mask = record["mask"][0]
        prediction = record["prediction"][0]

        axes[row, 0].imshow(image.permute(1, 2, 0))
        axes[row, 0].set_title(
            f"{group} #{record['index']}\n"
            f"Dice={record['dice']:.4f}, "
            f"IoU={record['iou']:.4f}"
        )

        axes[row, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title("Ground truth")

        axes[row, 2].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title("Prediction")

        axes[row, 3].imshow(image.permute(1, 2, 0))
        axes[row, 3].imshow(
            prediction,
            cmap="Reds",
            alpha=0.4,
            vmin=0,
            vmax=1,
        )
        axes[row, 3].set_title("Prediction overlay")

        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"找不到最佳模型：{CHECKPOINT_PATH}"
        )

    device = torch.device("cuda")

    test_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="test",
        training=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = create_model(device)

    print("device:", device)
    print("test size:", len(test_dataset))
    print("threshold:", THRESHOLD)

    global_dice, global_iou, records = evaluate(
        model,
        test_loader,
        device,
    )

    dice_values = np.array(
        [record["dice"] for record in records]
    )

    iou_values = np.array(
        [record["iou"] for record in records]
    )

    metrics = {
        "checkpoint": str(CHECKPOINT_PATH),
        "test_size": len(test_dataset),
        "threshold": THRESHOLD,
        "global_dice": float(global_dice),
        "global_iou": float(global_iou),
        "mean_per_image_dice": float(dice_values.mean()),
        "median_per_image_dice": float(np.median(dice_values)),
        "minimum_per_image_dice": float(dice_values.min()),
        "maximum_per_image_dice": float(dice_values.max()),
        "mean_per_image_iou": float(iou_values.mean()),
        "median_per_image_iou": float(np.median(iou_values)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "test_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    cases = select_cases(records)
    save_cases(cases, ASSET_PATH)

    print("global dice:", metrics["global_dice"])
    print("global iou:", metrics["global_iou"])
    print(
        "mean per-image dice:",
        metrics["mean_per_image_dice"],
    )
    print(
        "median per-image dice:",
        metrics["median_per_image_dice"],
    )
    print(
        "minimum per-image dice:",
        metrics["minimum_per_image_dice"],
    )
    print(
        "maximum per-image dice:",
        metrics["maximum_per_image_dice"],
    )
    print("metrics:", metrics_path)
    print("cases:", ASSET_PATH)
    print("test evaluation: FINISHED")


if __name__ == "__main__":
    main()