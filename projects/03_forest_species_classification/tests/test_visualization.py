import torch

from src.visualization import denormalize_imagenet


def test_denormalize_imagenet_recovers_unit_rgb_values():
    original = torch.tensor([
        [[0.0, 1.0]],
        [[0.25, 0.75]],
        [[0.5, 0.1]],
    ])
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    normalized = (original - mean) / std

    recovered = denormalize_imagenet(normalized)

    assert torch.allclose(recovered, original, atol=1e-6)


def test_denormalize_imagenet_clamps_values_for_display():
    normalized = torch.full((3, 2, 2), 100.0)
    recovered = denormalize_imagenet(normalized)
    assert recovered.min().item() >= 0.0
    assert recovered.max().item() <= 1.0
