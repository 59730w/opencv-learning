import csv
from pathlib import Path

import cv2
import numpy as np


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

INVENTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "data_audit"
    / "image_inventory.csv"
)

EXCLUSIONS_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "exclusions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "data_audit"
    / "perceptual_hashes.csv"
)


def calculate_phash(image_path: Path) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    resized = cv2.resize(
        image,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    resized = np.float32(resized)
    dct = cv2.dct(resized)

    low_frequency = dct[:8, :8].copy()

    # 不使用代表整体亮度的直流分量
    values = low_frequency.flatten()
    median = np.median(values[1:])

    bits = values > median

    bit_string = "".join(
        "1" if bit else "0"
        for bit in bits
    )

    return f"{int(bit_string, 2):016x}"


def calculate_dhash(image_path: Path) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    resized = cv2.resize(
        image,
        (9, 8),
        interpolation=cv2.INTER_AREA,
    )

    differences = resized[:, 1:] > resized[:, :-1]

    bit_string = "".join(
        "1" if bit else "0"
        for bit in differences.flatten()
    )

    return f"{int(bit_string, 2):016x}"


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
        inventory = list(csv.DictReader(file))

    eligible_records = [
        row
        for row in inventory
        if row["relative_path"] not in excluded_paths
    ]

    results = []

    for index, record in enumerate(eligible_records, start=1):
        relative_path = record["relative_path"]
        image_path = DATA_ROOT / relative_path

        results.append({
            "relative_path": relative_path,
            "class_name": record["class_name"],
            "sha256": record["sha256"],
            "phash": calculate_phash(image_path),
            "dhash": calculate_dhash(image_path),
        })

        if index % 500 == 0 or index == len(eligible_records):
            print(f"已计算 {index}/{len(eligible_records)}")

    if len(results) != 5558:
        raise ValueError(
            f"预期计算5558张，实际为{len(results)}张"
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "class_name",
                "sha256",
                "phash",
                "dhash",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"感知哈希记录数：{len(results)}")
    print(f"结果已保存：{OUTPUT_PATH}")
    print("本步骤没有修改或删除原始图片")


if __name__ == "__main__":
    main()