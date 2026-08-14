import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

INPUT_PATH = AUDIT_DIR / "exact_duplicate_conflicts.csv"
OUTPUT_PATH = PROCESSED_DIR / "exclusions.csv"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    exclusions = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            exclusions.append({
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "sha256": row["sha256"],
                "reason": "exact duplicate with conflicting class label",
                "action": "exclude from train, validation and test",
            })

    if len(exclusions) != 20:
        raise ValueError(
            f"预期20条排除记录，实际为{len(exclusions)}条"
        )

    paths = [row["relative_path"] for row in exclusions]
    if len(paths) != len(set(paths)):
        raise ValueError("排除清单中存在重复路径")

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
                "reason",
                "action",
            ],
        )
        writer.writeheader()
        writer.writerows(exclusions)

    print(f"排除记录数：{len(exclusions)}")
    print("原始图片未删除或修改")
    print(f"排除清单已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
