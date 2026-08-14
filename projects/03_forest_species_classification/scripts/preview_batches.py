import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ManifestImageDataset
from src.transforms import build_eval_transform, build_train_transform
from src.visualization import denormalize_imagenet


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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_audit"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_batch(split, transform, output_name, seed):
    set_seed(seed)
    dataset = ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split=split,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=(split == "train"),
        num_workers=0,
    )
    images, labels, paths = next(iter(loader))
    idx_to_class = {
        index: name
        for name, index in dataset.class_to_idx.items()
    }

    figure, axes = plt.subplots(4, 4, figsize=(14, 14))
    for axis, image, label, relative_path in zip(
        axes.flatten(), images, labels, paths
    ):
        display = denormalize_imagenet(image).permute(1, 2, 0).numpy()
        axis.imshow(display)
        axis.set_title(
            f"{idx_to_class[int(label)]}\n{Path(relative_path).name}",
            fontsize=8,
        )
        axis.axis("off")

    figure.suptitle(f"{split} batch — {len(dataset)} images", fontsize=16)
    figure.tight_layout()
    output_path = OUTPUT_DIR / output_name
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return {
        "split": split,
        "dataset_size": len(dataset),
        "batch_shape": list(images.shape),
        "label_min": int(labels.min()),
        "label_max": int(labels.max()),
        "output_path": str(output_path),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [
        save_batch(
            "train",
            build_train_transform(),
            "train_batch.jpg",
            seed=42,
        ),
        save_batch(
            "validation",
            build_eval_transform(),
            "validation_batch.jpg",
            seed=42,
        ),
    ]
    summary_path = OUTPUT_DIR / "batch_preview_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for summary in summaries:
        print(
            f"{summary['split']}：{summary['dataset_size']}张，"
            f"批次形状={summary['batch_shape']}，"
            f"输出={summary['output_path']}"
        )
    print("原始图片未复制、移动或修改")


if __name__ == "__main__":
    main()
