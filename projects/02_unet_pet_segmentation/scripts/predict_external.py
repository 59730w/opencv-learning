from pathlib import Path
import math

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "external_images"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "baseline" / "best_model.pth"
)
OUTPUT_PATH = PROJECT_ROOT / "assets" / "external_predictions.png"

IMAGE_SIZE = 256
THRESHOLD = 0.5
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


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


def letterbox(image):
    """保持纵横比缩放，并用黑色补边到256×256。"""
    original_width, original_height = image.size

    scale = min(
        IMAGE_SIZE / original_width,
        IMAGE_SIZE / original_height,
    )

    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    right = IMAGE_SIZE - resized_width - left
    bottom = IMAGE_SIZE - resized_height - top

    padded = ImageOps.expand(
        resized,
        border=(left, top, right, bottom),
        fill=(0, 0, 0),
    )

    return padded, (left, top, right, bottom)


def preprocess(image):
    padded, padding = letterbox(image)

    tensor = TF.pil_to_tensor(padded).float() / 255.0
    normalized = (
        tensor - IMAGENET_MEAN
    ) / IMAGENET_STD

    return padded, normalized.unsqueeze(0), padding


def remove_padding_and_restore(mask, padding, original_size):
    left, top, right, bottom = padding

    width_end = IMAGE_SIZE - right if right > 0 else IMAGE_SIZE
    height_end = IMAGE_SIZE - bottom if bottom > 0 else IMAGE_SIZE

    mask = mask[top:height_end, left:width_end]

    mask_image = Image.fromarray(
        (mask * 255).to(torch.uint8).numpy()
    )

    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST,
    )

    return mask_image


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"外部图片目录不存在：{INPUT_DIR}"
        )

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not image_paths:
        raise RuntimeError("external_images中没有可用图片")

    device = torch.device("cuda")
    model = create_model(device)

    results = []

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")

            _, tensor, padding = preprocess(image)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(tensor.to(device))

            probability = torch.sigmoid(logits)[0, 0].cpu()
            binary_mask = (probability >= THRESHOLD).float()

            restored_mask = remove_padding_and_restore(
                binary_mask,
                padding,
                image.size,
            )

            foreground_ratio = (
                torch.from_numpy(
                    __import__("numpy").array(restored_mask)
                ) > 0
            ).float().mean().item()

            results.append(
                {
                    "path": image_path,
                    "image": image,
                    "mask": restored_mask,
                    "foreground_ratio": foreground_ratio,
                }
            )

            print(
                image_path.name,
                f"foreground ratio={foreground_ratio:.4f}",
            )

    rows = len(results)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))

    if rows == 1:
        axes = axes.reshape(1, 3)

    for row, result in enumerate(results):
        axes[row, 0].imshow(result["image"])
        axes[row, 0].set_title(result["path"].name)

        axes[row, 1].imshow(
            result["mask"],
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[row, 1].set_title(
            f"Prediction mask\n"
            f"FG={result['foreground_ratio']:.3f}"
        )

        axes[row, 2].imshow(result["image"])
        axes[row, 2].imshow(
            result["mask"],
            cmap="Reds",
            alpha=0.4,
            vmin=0,
            vmax=255,
        )
        axes[row, 2].set_title("Prediction overlay")

        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("images:", len(results))
    print("threshold:", THRESHOLD)
    print("saved:", OUTPUT_PATH)
    print("external test: FINISHED")


if __name__ == "__main__":
    main()