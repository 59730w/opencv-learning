import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2


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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_audit"


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATA_ROOT.rglob("*.JPG"))
    records = []
    unreadable = []

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            unreadable.append(str(image_path))
            continue

        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        records.append({
            "relative_path": image_path.relative_to(DATA_ROOT).as_posix(),
            "class_name": image_path.parent.name,
            "width": width,
            "height": height,
            "channels": channels,
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "blur_score": round(blur_score, 4),
            "sha256": calculate_sha256(image_path),
        })

        if index % 500 == 0 or index == len(image_paths):
            print(f"已处理 {index}/{len(image_paths)}")

    inventory_path = OUTPUT_DIR / "image_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    class_counts = Counter(record["class_name"] for record in records)
    size_counts = Counter(
        f"{record['width']}x{record['height']}" for record in records
    )

    summary = {
        "data_root": str(DATA_ROOT),
        "discovered_images": len(image_paths),
        "readable_images": len(records),
        "unreadable_images": unreadable,
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "size_counts": dict(sorted(size_counts.items())),
        "channel_counts": dict(
            Counter(record["channels"] for record in records)
        ),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"清单已保存：{inventory_path}")
    print(f"摘要已保存：{summary_path}")
    print(f"可读取图片：{len(records)}")
    print(f"无法读取图片：{len(unreadable)}")
    print(f"类别数量：{len(class_counts)}")


if __name__ == "__main__":
    main()