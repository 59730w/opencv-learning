import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
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
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "resnet18_baseline"
    / "test_predictions.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "resnet18_baseline"
    / "misclassified_contact_sheet.jpg"
)


def main():
    with PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    errors = [
        row
        for row in rows
        if row["correct"].lower() == "false"
    ]

    # 高置信度错误排在前面
    errors.sort(
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )

    assert len(errors) == 16

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(16, 17),
    )
    axes = axes.flatten()

    for index, (axis, row) in enumerate(zip(axes, errors), start=1):
        image_path = DATA_ROOT / row["relative_path"]
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"OpenCV无法读取：{image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        confidence = float(row["confidence"])

        axis.imshow(image)
        axis.set_title(
            f"{index}. True: {row['true_class']}\n"
            f"Pred: {row['predicted_class']} | "
            f"Conf: {confidence:.1%}",
            fontsize=8,
            color="#8A4B08",
        )
        axis.axis("off")

    fig.suptitle(
        "Misclassified BarkVN-50 test images",
        fontsize=18,
    )
    fig.text(
        0.5,
        0.965,
        "16 errors in 823 test images · sorted by prediction confidence",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("误分类图片:", len(errors))
    print("高置信度错误：")

    for row in errors:
        print(
            f"{float(row['confidence']):.2%} | "
            f"{row['true_class']} → "
            f"{row['predicted_class']} | "
            f"{row['relative_path']}"
        )

    print("误分类联系表:", OUTPUT_PATH)

    assert OUTPUT_PATH.is_file()
    assert OUTPUT_PATH.stat().st_size > 0

    print("第五天第四步验证成功")
    print("本步骤没有修改、移动或删除原始图片")


if __name__ == "__main__":
    main()