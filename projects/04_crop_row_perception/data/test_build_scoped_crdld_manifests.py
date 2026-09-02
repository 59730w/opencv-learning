from pathlib import Path
import sys


DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from build_scoped_crdld_manifests import build_scoped_manifests


def _pair(root: Path, split: str, stem: str, image_bytes: bytes) -> None:
    image_dir = root / split / "image"
    label_dir = root / split / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(image_bytes)
    (label_dir / f"{stem}.jpg").write_bytes(b"label-" + image_bytes)


def test_build_scoped_manifests_excludes_benchmark_duplicate(tmp_path: Path) -> None:
    _pair(tmp_path, "train_data", "1", b"train-one")
    _pair(tmp_path, "train_data", "2", b"shared-image")
    _pair(tmp_path, "validation_data", "1", b"validation-one")
    _pair(tmp_path, "test_data", "1", b"benchmark-one")
    _pair(tmp_path, "test_data", "2", b"shared-image")

    result = build_scoped_manifests(tmp_path, tmp_path / "manifests")

    assert result["counts"] == {
        "train_development": 2,
        "validation_development": 1,
        "same_source_internal_benchmark": 1,
    }
    assert result["excluded_cross_role_duplicates"] == ["test_data/2"]
    assert result["scoped_gate"] == "PASS"
    assert result["full_reject_aware_gate"] == "BLOCKED"
