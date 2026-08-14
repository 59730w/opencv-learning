import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"
INVENTORY_PATH = AUDIT_DIR / "image_inventory.csv"


def main():
    hash_groups = defaultdict(list)

    with INVENTORY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            hash_groups[row["sha256"]].append({
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
            })

    duplicate_groups = []
    same_class_groups = 0
    cross_class_groups = 0

    for sha256, images in sorted(hash_groups.items()):
        if len(images) < 2:
            continue

        class_names = sorted({
            image["class_name"]
            for image in images
        })

        is_cross_class = len(class_names) > 1

        if is_cross_class:
            cross_class_groups += 1
        else:
            same_class_groups += 1

        duplicate_groups.append({
            "sha256": sha256,
            "image_count": len(images),
            "class_names": class_names,
            "cross_class": is_cross_class,
            "images": images,
        })

    duplicate_image_count = sum(
        group["image_count"]
        for group in duplicate_groups
    )

    removable_copy_count = sum(
        group["image_count"] - 1
        for group in duplicate_groups
    )

    result = {
        "total_inventory_images": sum(
            len(images)
            for images in hash_groups.values()
        ),
        "unique_sha256_count": len(hash_groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_image_count": duplicate_image_count,
        "removable_copy_count": removable_copy_count,
        "same_class_duplicate_groups": same_class_groups,
        "cross_class_duplicate_groups": cross_class_groups,
        "duplicate_groups": duplicate_groups,
    }

    output_path = AUDIT_DIR / "exact_duplicates.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"清单图片总数：{result['total_inventory_images']}")
    print(f"唯一 SHA-256 数量：{result['unique_sha256_count']}")
    print(f"完全重复组数：{result['duplicate_group_count']}")
    print(f"重复组涉及图片数：{duplicate_image_count}")
    print(f"多余副本数量：{removable_copy_count}")
    print(f"同类别重复组：{same_class_groups}")
    print(f"跨类别重复组：{cross_class_groups}")
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()