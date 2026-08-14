import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ManifestImageDataset
from src.transforms import build_eval_transform


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
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "data_audit" / "pipeline_validation.json"
EXPECTED_COUNTS = {"train": 3891, "validation": 844, "test": 823}


def validate_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    paths = [row["relative_path"] for row in rows]
    group_splits = defaultdict(set)
    class_splits = defaultdict(set)
    split_counts = Counter()

    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
        class_splits[row["class_name"]].add(row["split"])
        split_counts[row["split"]] += 1

    result = {
        "total_rows": len(rows),
        "unique_paths": len(set(paths)),
        "split_counts": dict(split_counts),
        "group_leakage_count": sum(
            len(splits) > 1 for splits in group_splits.values()
        ),
        "classes_missing_a_split": sum(
            splits != set(EXPECTED_COUNTS) for splits in class_splits.values()
        ),
        "missing_files": sum(not (DATA_ROOT / path).is_file() for path in paths),
    }

    assert result["total_rows"] == 5558
    assert result["unique_paths"] == 5558
    assert result["split_counts"] == EXPECTED_COUNTS
    assert result["group_leakage_count"] == 0
    assert result["classes_missing_a_split"] == 0
    assert result["missing_files"] == 0
    return result


def validate_all_images():
    results = {}
    transform = build_eval_transform()

    for split, expected_count in EXPECTED_COUNTS.items():
        dataset = ManifestImageDataset(
            data_root=DATA_ROOT,
            manifest_path=MANIFEST_PATH,
            class_map_path=CLASS_MAP_PATH,
            split=split,
            transform=transform,
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        loaded = 0

        for images, labels, paths in loader:
            assert images.ndim == 4 and tuple(images.shape[1:]) == (3, 224, 224)
            assert images.dtype == torch.float32
            assert torch.isfinite(images).all()
            assert int(labels.min()) >= 0 and int(labels.max()) < 50
            assert len(paths) == images.shape[0]
            loaded += images.shape[0]

        assert loaded == expected_count
        results[split] = loaded

    return results


def main():
    summary = {
        "manifest": validate_manifest(),
        "fully_loaded_images": validate_all_images(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"清单总行数：{summary['manifest']['total_rows']}")
    print(f"唯一图片路径：{summary['manifest']['unique_paths']}")
    print(f"分组跨子集泄漏：{summary['manifest']['group_leakage_count']}")
    print(f"未覆盖三个子集的类别：{summary['manifest']['classes_missing_a_split']}")
    print(f"缺失文件：{summary['manifest']['missing_files']}")
    print(f"全量成功加载：{summary['fully_loaded_images']}")
    print(f"验证结果：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
