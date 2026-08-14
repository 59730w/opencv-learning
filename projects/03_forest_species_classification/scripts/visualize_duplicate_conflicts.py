import csv
from collections import defaultdict
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
CSV_PATH = AUDIT_DIR / "exact_duplicate_conflicts.csv"
OUTPUT_PATH = AUDIT_DIR / "exact_duplicate_conflicts.jpg"


def load_rgb(path: Path):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    groups = defaultdict(list)

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            groups[int(row["group_number"])].append(row)

    group_numbers = sorted(groups)
    figure, axes = plt.subplots(
        len(group_numbers),
        2,
        figsize=(8, 3 * len(group_numbers)),
    )

    for row_index, group_number in enumerate(group_numbers):
        images = sorted(
            groups[group_number],
            key=lambda item: item["class_name"],
        )

        if len(images) != 2:
            raise ValueError(
                f"重复组 {group_number} 预期有 2 张图片，实际为 {len(images)}"
            )

        for column_index, record in enumerate(images):
            image_path = DATA_ROOT / record["relative_path"]
            image = load_rgb(image_path)

            axis = axes[row_index, column_index]
            axis.imshow(image)
            axis.set_title(
                f"Group {group_number}\n"
                f"{record['class_name']}\n"
                f"{image_path.name}",
                fontsize=9,
            )
            axis.axis("off")

    figure.suptitle(
        "Cross-class exact duplicate conflicts",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"冲突组数：{len(group_numbers)}")
    print(f"可视化已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
