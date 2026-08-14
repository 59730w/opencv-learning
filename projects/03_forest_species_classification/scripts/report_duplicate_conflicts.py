import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"

INPUT_PATH = AUDIT_DIR / "exact_duplicates.json"
OUTPUT_PATH = AUDIT_DIR / "exact_duplicate_conflicts.csv"


def main():
    result = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    rows = []
    cross_class_groups = []

    for group_number, group in enumerate(
        result["duplicate_groups"],
        start=1,
    ):
        if not group["cross_class"]:
            continue

        cross_class_groups.append(group)
        print(f"\n重复组 {group_number}")
        print(f"SHA-256：{group['sha256']}")

        for image in group["images"]:
            print(
                f"  类别：{image['class_name']}\n"
                f"  路径：{image['relative_path']}"
            )

            rows.append({
                "group_number": group_number,
                "sha256": group["sha256"],
                "class_name": image["class_name"],
                "relative_path": image["relative_path"],
            })

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group_number",
                "sha256",
                "class_name",
                "relative_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n跨类别冲突组数：{len(cross_class_groups)}")
    print(f"冲突图片记录数：{len(rows)}")
    print(f"结果已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
