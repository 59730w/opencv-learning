from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_class_names(class_map_path: Path) -> list[str]:
    with class_map_path.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)

    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("class map must be a non-empty JSON object")
    if any(not isinstance(index, int) for index in class_to_idx.values()):
        raise ValueError("class indices must be integers")

    expected_indices = list(range(len(class_to_idx)))
    if sorted(class_to_idx.values()) != expected_indices:
        raise ValueError("class indices must be continuous from zero")

    class_names = [""] * len(class_to_idx)
    for class_name, class_index in class_to_idx.items():
        class_names[class_index] = class_name
    return class_names


def build_model_from_checkpoint(
    checkpoint_path: Path,
    class_map_path: Path,
) -> tuple[nn.Module, list[str], dict]:
    class_names = read_class_names(class_map_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    configured_classes = checkpoint.get("config", {}).get("num_classes")
    if configured_classes != len(class_names):
        raise ValueError(
            "checkpoint class count does not match class_to_idx.json: "
            f"{configured_classes} != {len(class_names)}"
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, class_names, checkpoint


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image_batch(image_paths: list[Path]) -> torch.Tensor:
    if not image_paths:
        raise ValueError("at least one image is required")

    transform = build_eval_transform()
    tensors = []
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        with Image.open(image_path) as image:
            tensors.append(transform(image.convert("RGB")))

    batch = torch.stack(tensors, dim=0)
    if batch.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {batch.dtype}")
    return batch
