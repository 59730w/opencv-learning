import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"

INVENTORY_PATH = AUDIT_DIR / "image_inventory.csv"
EXCLUSIONS_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "exclusions.csv"
)
OUTLIERS_PATH = AUDIT_DIR / "quality_outliers.csv"
SUMMARY_PATH = AUDIT_DIR / "quality_summary.json"
FIGURE_PATH = AUDIT_DIR / "quality_distributions.png"


def percentile(values, percentage):
    return float(np.percentile(values, percentage))


def main():
    with EXCLUSIONS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        excluded_paths = {
            row["relative_path"]
            for row in csv.DictReader(file)
        }

    with INVENTORY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        records = [
            row
            for row in csv.DictReader(file)
            if row["relative_path"] not in excluded_paths
        ]

    for row in records:
        for key in (
            "width",
            "height",
            "channels",
            "brightness",
            "contrast",
            "blur_score",
        ):
            row[key] = float(row[key])

    brightness = np.array([
        row["brightness"] for row in records
    ])
    contrast = np.array([
        row["contrast"] for row in records
    ])
    blur = np.array([
        row["blur_score"] for row in records
    ])

    thresholds = {
        "brightness_low_p1": percentile(brightness, 1),
        "brightness_high_p99": percentile(brightness, 99),
        "contrast_low_p1": percentile(contrast, 1),
        "blur_low_p1": percentile(blur, 1),
    }

    outliers = []

    for row in records:
        reasons = []

        if row["brightness"] <= thresholds["brightness_low_p1"]:
            reasons.append("very_dark")

        if row["brightness"] >= thresholds["brightness_high_p99"]:
            reasons.append("very_bright")

        if row["contrast"] <= thresholds["contrast_low_p1"]:
            reasons.append("low_contrast")

        if row["blur_score"] <= thresholds["blur_low_p1"]:
            reasons.append("low_laplacian_variance")

        if (int(row["width"]), int(row["height"])) not in {
            (303, 404),
            (245, 327),
        }:
            reasons.append("uncommon_size")

        if reasons:
            outliers.append({
                **row,
                "reasons": "|".join(reasons),
            })

    with OUTLIERS_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = list(outliers[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outliers)

    class_counts = Counter(
        row["class_name"] for row in records
    )
    size_counts = Counter(
        f"{int(row['width'])}x{int(row['height'])}"
        for row in records
    )
    reason_counts = Counter()

    for row in outliers:
        reason_counts.update(row["reasons"].split("|"))

    summary = {
        "eligible_images": len(records),
        "class_count": len(class_counts),
        "minimum_class_count": min(class_counts.values()),
        "maximum_class_count": max(class_counts.values()),
        "thresholds": thresholds,
        "size_counts": dict(size_counts),
        "outlier_image_count": len(outliers),
        "outlier_reason_counts": dict(reason_counts),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].bar(
        range(len(class_counts)),
        [class_counts[name] for name in sorted(class_counts)],
    )
    axes[0, 0].set_title("Images per class")
    axes[0, 0].set_xlabel("Class index")
    axes[0, 0].set_ylabel("Image count")

    axes[0, 1].hist(brightness, bins=40)
    axes[0, 1].set_title("Brightness distribution")

    axes[1, 0].hist(contrast, bins=40)
    axes[1, 0].set_title("Contrast distribution")

    axes[1, 1].hist(np.log1p(blur), bins=40)
    axes[1, 1].set_title("log(1 + Laplacian variance)")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)

    print(f"参与统计图片：{len(records)}")
    print(f"类别数量：{len(class_counts)}")
    print(
        f"每类最少/最多："
        f"{min(class_counts.values())}/"
        f"{max(class_counts.values())}"
    )
    print(f"异常候选图片：{len(outliers)}")
    print(f"异常原因统计：{dict(reason_counts)}")
    print(f"统计图：{FIGURE_PATH}")
    print(f"异常清单：{OUTLIERS_PATH}")


if __name__ == "__main__":
    main()