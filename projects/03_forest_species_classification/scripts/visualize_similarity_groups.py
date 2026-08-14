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

GROUPS_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "similarity_groups.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "data_audit"
    / "similarity_groups"
)

MAX_GROUPS = 20


def load_rgb(relative_path):
    image_path = DATA_ROOT / relative_path
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_group(group_id, records):
    columns = 4
    rows = math.ceil(len(records) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 4),
    )

    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for axis in axes:
        axis.axis("off")

    class_names = sorted({
        record["class_name"]
        for record in records
    })

    for axis, record in zip(axes, records):
        relative_path = record["relative_path"]
        axis.imshow(load_rgb(relative_path))
        axis.set_title(
            f"{record['class_name']}\n"
            f"{Path(relative_path).name}",
            fontsize=9,
        )
        axis.axis("off")

    figure.suptitle(
        f"{group_id} — {len(records)} images\n"
        f"Classes: {', '.join(class_names)}",
        fontsize=14,
    )
    figure.tight_layout()

    output_path = OUTPUT_DIR / f"{group_id}.jpg"
    figure.savefig(
        output_path,
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def main():
    groups = defaultdict(list)

    with GROUPS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for record in csv.DictReader(file):
            if record["is_similarity_group"].lower() == "true":
                groups[record["group_id"]].append(record)

    selected_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
        ),
    )[:MAX_GROUPS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for group_id, records in selected_groups:
        records.sort(key=lambda row: row["relative_path"])
        output_path = save_group(group_id, records)

        print(
            f"{group_id}：{len(records)}张，"
            f"类别：{records[0]['class_name']} → {output_path}"
        )

    print(f"已可视化最大相似组：{len(selected_groups)}个")
    print("本步骤没有排除、删除或修改图片")


if __name__ == "__main__":
    main()