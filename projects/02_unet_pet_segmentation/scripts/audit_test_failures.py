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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "failure_audit"
ASSET_PATH = PROJECT_ROOT / "assets" / "valid_failure_cases.png"

BATCH_SIZE = 4
THRESHOLD = 0.5
MIN_VALID_FOREGROUND_RATIO = 0.01
MAX_VALID_FOREGROUND_RATIO = 0.99
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

    return model


def collect_records(model, loader, dataset, device):
    records = []
    dataset_index = 0

    with torch.no_grad():
        for images, masks in loader:
            images_device = images.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(images_device)

            predictions = (
                torch.sigmoid(logits).cpu() >= THRESHOLD
            ).float()

            for position in range(images.size(0)):
                mask = masks[position]
                prediction = predictions[position]

                intersection = (prediction * mask).sum().item()
                prediction_sum = prediction.sum().item()
                target_sum = mask.sum().item()
                union = ((prediction + mask) > 0).sum().item()

                dice = (2 * intersection + EPSILON) / (
                    prediction_sum + target_sum + EPSILON
                )

                iou = (intersection + EPSILON) / (
                    union + EPSILON
                )

                foreground_ratio = target_sum / mask.numel()

                image_path = Path(
                    dataset.dataset._images[dataset_index]
                )

                mask_path = Path(
                    dataset.dataset._segs[dataset_index]
                )

                records.append(
                    {
                        "index": dataset_index,
                        "filename": image_path.name,
                        "mask_filename": mask_path.name,
                        "foreground_ratio": foreground_ratio,
                        "dice": dice,
                        "iou": iou,
                        "image": images[position],
                        "mask": mask,
                        "prediction": prediction,
                    }
                )

                dataset_index += 1

    return records


def serializable_record(record):
    return {
        "index": record["index"],
        "filename": record["filename"],
        "mask_filename": record["mask_filename"],
        "foreground_ratio": record["foreground_ratio"],
        "dice": record["dice"],
        "iou": record["iou"],
    }


def save_valid_failures(records, output_path):
    valid_records = [
        record
        for record in records
        if (
            MIN_VALID_FOREGROUND_RATIO
            <= record["foreground_ratio"]
            <= MAX_VALID_FOREGROUND_RATIO
        )
    ]

    worst_valid = sorted(
        valid_records,
        key=lambda record: record["dice"],
    )[:9]

    fig, axes = plt.subplots(9, 4, figsize=(12, 27))

    for row, record in enumerate(worst_valid):
        image = (
            record["image"] * IMAGENET_STD
            + IMAGENET_MEAN
        ).clamp(0, 1)

        mask = record["mask"][0]
        prediction = record["prediction"][0]

        axes[row, 0].imshow(image.permute(1, 2, 0))
        axes[row, 0].set_title(
            f"#{record['index']} {record['filename']}\n"
            f"Dice={record['dice']:.4f}, "
            f"FG={record['foreground_ratio']:.3f}"
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

    return worst_valid


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

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

    records = collect_records(
        model,
        test_loader,
        test_dataset,
        device,
    )

    all_background = [
        record
        for record in records
        if record["foreground_ratio"] == 0
    ]

    near_background = [
        record
        for record in records
        if (
            0 < record["foreground_ratio"]
            < MIN_VALID_FOREGROUND_RATIO
        )
    ]

    near_foreground = [
        record
        for record in records
        if (
            record["foreground_ratio"]
            > MAX_VALID_FOREGROUND_RATIO
        )
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    worst_valid = save_valid_failures(records, ASSET_PATH)

    report = {
        "test_size": len(records),
        "valid_foreground_range": [
            MIN_VALID_FOREGROUND_RATIO,
            MAX_VALID_FOREGROUND_RATIO,
        ],
        "all_background_count": len(all_background),
        "near_background_count": len(near_background),
        "near_foreground_count": len(near_foreground),
        "all_background_samples": [
            serializable_record(record)
            for record in all_background
        ],
        "near_background_samples": [
            serializable_record(record)
            for record in near_background
        ],
        "near_foreground_samples": [
            serializable_record(record)
            for record in near_foreground
        ],
        "worst_valid_samples": [
            serializable_record(record)
            for record in worst_valid
        ],
    }

    report_path = OUTPUT_DIR / "failure_audit.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("test size:", len(records))
    print("all-background masks:", len(all_background))
    print("near-background masks:", len(near_background))
    print("near-foreground masks:", len(near_foreground))

    print("\nall-background samples:")
    for record in all_background:
        print(
            record["index"],
            record["filename"],
            f"dice={record['dice']:.4f}",
        )

    print("\nworst valid samples:")
    for record in worst_valid:
        print(
            record["index"],
            record["filename"],
            f"dice={record['dice']:.4f}",
            f"fg={record['foreground_ratio']:.4f}",
        )

    print("\nreport:", report_path)
    print("figure:", ASSET_PATH)
    print("failure audit: FINISHED")


if __name__ == "__main__":
    main()