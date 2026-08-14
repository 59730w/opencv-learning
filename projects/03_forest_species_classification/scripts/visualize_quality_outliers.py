import csv
import math
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
INPUT_PATH = AUDIT_DIR / "quality_outliers.csv"

REASONS = [
    "very_dark",
    "very_bright",
    "low_contrast",
    "low_laplacian_variance",
    "uncommon_size",
]


def load_rgb(image_path: Path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_contact_sheet(reason, records):
    columns = 4
    rows = math.ceil(len(records) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 4),
    )

    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis in axes:
        axis.axis("off")

    for axis, record in zip(axes, records):
        image_path = DATA_ROOT / record["relative_path"]
        image = load_rgb(image_path)

        axis.imshow(image)
        axis.set_title(
            f"{record['class_name']}\n"
            f"{image_path.name}\n"
            f"{int(float(record['width']))}x"
            f"{int(float(record['height']))}\n"
            f"B={float(record['brightness']):.1f}, "
            f"C={float(record['contrast']):.1f}\n"
            f"Blur={float(record['blur_score']):.1f}",
            fontsize=8,
        )
        axis.axis("off")

    figure.suptitle(
        f"{reason} — {len(records)} images",
        fontsize=16,
    )
    figure.tight_layout()

    output_path = AUDIT_DIR / f"outliers_{reason}.jpg"
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)

    print(f"{reason}：{len(records)}张 → {output_path}")


def main():
    grouped_records = defaultdict(list)

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for record in csv.DictReader(file):
            for reason in record["reasons"].split("|"):
                grouped_records[reason].append(record)

    for reason in REASONS:
        records = grouped_records.get(reason, [])

        if not records:
            print(f"{reason}：没有候选图片")
            continue

        records.sort(key=lambda row: row["relative_path"])
        create_contact_sheet(reason, records)

    print("异常候选联系表生成完成")
    print("本步骤没有删除、移动或修改原始图片")


if __name__ == "__main__":
    main()