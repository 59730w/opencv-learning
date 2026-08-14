import importlib
import importlib.util

import torch
from PIL import Image

from src import transforms as transform_module


def test_five_crop_eval_transform_returns_five_model_inputs():
    assert hasattr(transform_module, "build_five_crop_eval_transform")

    transform = transform_module.build_five_crop_eval_transform()
    image = Image.new("RGB", (480, 320), color=(120, 90, 60))
    crops = transform(image)

    assert crops.shape == (5, 3, 224, 224)
    assert crops.dtype == torch.float32


def test_average_crop_probabilities_averages_softmax_over_crops():
    assert importlib.util.find_spec("src.inference") is not None
    inference_module = importlib.import_module("src.inference")

    logits = torch.tensor([
        [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ])
    probabilities = inference_module.average_crop_probabilities(logits)
    expected = torch.softmax(logits, dim=2).mean(dim=1)

    assert probabilities.shape == (2, 3)
    assert torch.allclose(probabilities, expected)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
