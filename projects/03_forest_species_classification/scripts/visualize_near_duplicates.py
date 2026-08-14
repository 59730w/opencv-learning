import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


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

AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"
INPUT_PATH = AUDIT_DIR / "near_duplicate_candidates.csv"
OUTPUT_PATH = AUDIT_DIR / "near_duplicate_top_pairs.jpg"

MAX_PAIRS = 40


def load_rgb(relative_path):
    image_path = DATA_ROOT / relative_path
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"无法读取：{image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        candidates = list(csv.DictReader(file))

    candidates.sort(
        key=lambda row: (
            int(row["phash_distance"])
            + int(row["dhash_distance"]),
            int(row["phash_distance"]),
            int(row["dhash_distance"]),
            row["first_path"],
            row["second_path"],
        )
    )

    selected = candidates[:MAX_PAIRS]

    rows = math.ceil(len(selected) / 2)
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(16, rows * 4),
    )
    axes = axes.reshape(rows, 4)

    for axis in axes.flatten():
        axis.axis("off")

    for index, record in enumerate(selected):
        grid_row = index // 2
        start_column = (index % 2) * 2

        pair = [
            (
                record["first_path"],
                record["first_class"],
            ),
            (
                record["second_path"],
                record["second_class"],
            ),
        ]

        for offset, (relative_path, class_name) in enumerate(pair):
            axis = axes[grid_row, start_column + offset]
            axis.imshow(load_rgb(relative_path))
            axis.set_title(
                f"Pair {index + 1}\n"
                f"{class_name}\n"
                f"{Path(relative_path).name}\n"
                f"p={record['phash_distance']}, "
                f"d={record['dhash_distance']}",
                fontsize=8,
            )
            axis.axis("off")

    figure.suptitle(
        "Top perceptual near-duplicate candidates",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_PATH,
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"候选总数：{len(candidates)}")
    print(f"本次可视化：{len(selected)}对")
    print(f"联系表：{OUTPUT_PATH}")
    print("本步骤没有排除、删除或修改图片")


if __name__ == "__main__":
    main()