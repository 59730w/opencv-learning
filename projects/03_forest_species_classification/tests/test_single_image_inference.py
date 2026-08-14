import json

import pytest
import torch
from PIL import Image

from src.inference import (
    prepare_rgb_image,
    read_indexed_class_names,
    topk_predictions,
)


def test_read_indexed_class_names_orders_classes_by_index(tmp_path):
    class_map_path = tmp_path / "class_to_idx.json"
    class_map_path.write_text(
        json.dumps({"second": 1, "first": 0, "third": 2}),
        encoding="utf-8",
    )

    assert read_indexed_class_names(class_map_path) == [
        "first",
        "second",
        "third",
    ]


def test_read_indexed_class_names_rejects_non_contiguous_indices(tmp_path):
    class_map_path = tmp_path / "class_to_idx.json"
    class_map_path.write_text(
        json.dumps({"first": 0, "third": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continuous"):
        read_indexed_class_names(class_map_path)


def test_prepare_rgb_image_converts_non_rgb_pil_image():
    grayscale = Image.new("L", (32, 24), color=128)

    converted = prepare_rgb_image(grayscale)

    assert converted.mode == "RGB"
    assert converted.size == (32, 24)


def test_prepare_rgb_image_rejects_missing_image():
    with pytest.raises(ValueError, match="image is required"):
        prepare_rgb_image(None)


def test_topk_predictions_returns_descending_named_probabilities():
    probabilities = torch.tensor([0.05, 0.70, 0.25])

    predictions = topk_predictions(
        probabilities,
        class_names=["A", "B", "C"],
        top_k=2,
    )

    assert [item.class_name for item in predictions] == ["B", "C"]
    assert [item.class_index for item in predictions] == [1, 2]
    assert predictions[0].probability == pytest.approx(0.70)
    assert predictions[1].probability == pytest.approx(0.25)


def test_topk_predictions_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        topk_predictions(
            torch.tensor([0.4, 0.6]),
            class_names=["A", "B"],
            top_k=0,
        )
