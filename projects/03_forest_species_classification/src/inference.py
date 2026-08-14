from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.model import build_resnet18
from src.transforms import build_eval_transform


@dataclass(frozen=True)
class Prediction:
    class_name: str
    class_index: int
    probability: float


def read_indexed_class_names(class_map_path):
    class_map_path = Path(class_map_path)
    with class_map_path.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)

    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("class map must be a non-empty object")

    indices = list(class_to_idx.values())
    if any(not isinstance(index, int) for index in indices):
        raise ValueError("class indices must be integers")

    expected = list(range(len(class_to_idx)))
    if sorted(indices) != expected:
        raise ValueError("class indices must be continuous from zero")

    class_names = [""] * len(class_to_idx)
    for class_name, class_index in class_to_idx.items():
        class_names[class_index] = class_name
    return class_names


def prepare_rgb_image(image):
    if image is None:
        raise ValueError("image is required")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    return image.convert("RGB")


def topk_predictions(probabilities, class_names, top_k=3):
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if len(class_names) != probabilities.numel():
        raise ValueError("class names and probabilities must have equal length")
    if not 1 <= top_k <= len(class_names):
        raise ValueError("top_k must be between one and the class count")

    top_probabilities, top_indices = torch.topk(probabilities, k=top_k)
    return [
        Prediction(
            class_name=class_names[class_index],
            class_index=class_index,
            probability=float(probability),
        )
        for probability, class_index in zip(
            top_probabilities.detach().cpu().tolist(),
            top_indices.detach().cpu().tolist(),
        )
    ]


class ForestSpeciesPredictor:
    def __init__(
        self,
        checkpoint_path,
        class_map_path,
        device: str | torch.device | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.class_map_path = Path(class_map_path)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.class_names = read_indexed_class_names(self.class_map_path)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        configured_classes = checkpoint.get("config", {}).get("num_classes")
        if configured_classes != len(self.class_names):
            raise ValueError(
                "checkpoint class count does not match the class mapping"
            )

        self.model = build_resnet18(
            num_classes=len(self.class_names),
            pretrained=False,
            freeze_backbone=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.transform = build_eval_transform()
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_stage = checkpoint.get("stage")

    def predict(self, image, top_k=3):
        rgb_image = prepare_rgb_image(image)
        model_input = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(model_input)
            probabilities = torch.softmax(logits.float(), dim=1).squeeze(0)

        return topk_predictions(
            probabilities=probabilities,
            class_names=self.class_names,
            top_k=top_k,
        )

    def predict_path(self, image_path, top_k=3):
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        with Image.open(image_path) as image:
            return self.predict(image, top_k=top_k)


def average_crop_probabilities(logits):
    if logits.ndim != 3:
        raise ValueError(
            "logits应为[batch, crops, classes]三维张量"
        )
    return torch.softmax(logits.float(), dim=2).mean(dim=1)
