from pathlib import Path
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pet_dataset import PetSegmentationDataset


DATA_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
ASSET_DIR = PROJECT_ROOT / "assets"

SEED = 42
VAL_RATIO = 0.2
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits, masks, epsilon=1e-7):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + masks.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + epsilon) / (
        denominator + epsilon
    )

    return 1 - dice.mean()


def combined_loss(logits, masks, bce_loss):
    return bce_loss(logits, masks) + dice_loss(logits, masks)


def create_datasets():
    # 训练集开启随机水平翻转
    training_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        training=True,
    )

    # 验证集关闭随机增强
    validation_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        training=False,
    )

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(training_dataset),
        generator=generator,
    ).tolist()

    validation_size = int(len(indices) * VAL_RATIO)
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    train_subset = Subset(
        training_dataset,
        training_indices,
    )

    val_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    return train_subset, val_subset, training_indices, validation_indices


RESNET18_WEIGHTS_URL = (
    "https://download.pytorch.org/models/"
    "resnet18-f37072fd.pth"
)


def create_model(device):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    encoder_state = torch.hub.load_state_dict_from_url(
        RESNET18_WEIGHTS_URL,
        map_location="cpu",
        progress=True,
        check_hash=True,
    )

    model.encoder.load_state_dict(encoder_state)

    return model.to(device)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    bce_loss,
    device,
):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = model(images)
            loss = combined_loss(logits, masks, bce_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, bce_loss, device):
    model.eval()

    total_loss = 0.0
    intersection = 0.0
    prediction_sum = 0.0
    target_sum = 0.0
    union = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(images)
                loss = combined_loss(logits, masks, bce_loss)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_loss += loss.item() * images.size(0)
            intersection += (predictions * masks).sum().item()
            prediction_sum += predictions.sum().item()
            target_sum += masks.sum().item()
            union += ((predictions + masks) > 0).sum().item()

    average_loss = total_loss / len(loader.dataset)

    dice = (2 * intersection) / (
        prediction_sum + target_sum + 1e-7
    )

    iou = intersection / (union + 1e-7)

    return average_loss, dice, iou


def save_curves(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        history["val_dice"],
        color="tab:green",
    )
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        epochs,
        history["val_iou"],
        color="tab:orange",
    )
    axes[2].set_title("Validation IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    device = torch.device("cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_indices, val_indices = (
        create_datasets()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = create_model(device)

    bce_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler("cuda")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "learning_rate": [],
    }

    best_dice = -1.0
    epochs_without_improvement = 0

    best_model_path = OUTPUT_DIR / "best_model.pth"
    history_path = OUTPUT_DIR / "history.json"
    split_path = OUTPUT_DIR / "split_indices.json"
    curve_path = ASSET_DIR / "baseline_training_curves.png"

    with split_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "seed": SEED,
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
            file,
            indent=2,
        )

    print("device:", device)
    print("train size:", len(train_dataset))
    print("val size:", len(val_dataset))
    print("batch size:", BATCH_SIZE)
    print("pretrained: ImageNet ResNet18")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            bce_loss,
            device,
        )

        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
            bce_loss,
            device,
        )

        scheduler.step(val_dice)

        learning_rate = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["learning_rate"].append(learning_rate)

        elapsed = time.time() - start_time

        print(
            f"epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val dice {val_dice:.4f} | "
            f"val iou {val_iou:.4f} | "
            f"lr {learning_rate:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": val_dice,
                    "val_iou": val_iou,
                    "seed": SEED,
                    "image_size": [256, 256],
                    "mask_mapping": {
                        "background_raw_2": 0,
                        "pet_raw_1_and_boundary_raw_3": 1,
                    },
                },
                best_model_path,
            )

            print("  saved new best model")
        else:
            epochs_without_improvement += 1

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        save_curves(history, curve_path)

        if epochs_without_improvement >= PATIENCE:
            print("early stopping")
            break

    print("best val dice:", best_dice)
    print("best model:", best_model_path)
    print("history:", history_path)
    print("curves:", curve_path)
    print("baseline training: FINISHED")


if __name__ == "__main__":
    main()