from pathlib import Path

import pytest
import torch
from torchvision.transforms import RandomRotation

from src.dataset import ManifestImageDataset
from src.transforms import build_eval_transform, build_train_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "BarkVN-50"
    / "v1"
    / "images"
    / "BarkVN-50_mendeley"
)
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "split_manifest.csv"
CLASS_MAP_PATH = PROJECT_ROOT / "datasets" / "processed" / "class_to_idx.json"

if not DATA_ROOT.is_dir():
    pytestmark = pytest.mark.skip(
        reason="BarkVN-50 image payload is not included in the Git repository"
    )


@pytest.fixture(scope="module")
def train_dataset():
    return ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split="train",
        transform=build_train_transform(),
    )


def test_manifest_dataset_has_expected_train_size(train_dataset):
    assert len(train_dataset) == 3891


def test_manifest_dataset_returns_image_label_and_path(train_dataset):
    image, label, relative_path = train_dataset[0]
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert 0 <= label < 50
    assert isinstance(relative_path, str)


def test_class_mapping_has_fifty_stable_indices(train_dataset):
    assert len(train_dataset.class_to_idx) == 50
    assert sorted(train_dataset.class_to_idx.values()) == list(range(50))


def test_eval_transform_is_deterministic():
    dataset = ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split="validation",
        transform=build_eval_transform(),
    )
    first, first_label, first_path = dataset[0]
    second, second_label, second_path = dataset[0]
    assert torch.equal(first, second)
    assert first_label == second_label
    assert first_path == second_path


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="未知数据子集"):
        ManifestImageDataset(
            data_root=DATA_ROOT,
            manifest_path=MANIFEST_PATH,
            class_map_path=CLASS_MAP_PATH,
            split="invalid",
            transform=build_eval_transform(),
        )


def test_train_rotation_uses_neutral_fill():
    transform = build_train_transform()
    rotations = [item for item in transform.transforms if isinstance(item, RandomRotation)]

    assert len(rotations) == 1
    assert tuple(rotations[0].fill) == (124, 116, 104)
