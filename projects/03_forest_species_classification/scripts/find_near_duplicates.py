import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"

INPUT_PATH = AUDIT_DIR / "perceptual_hashes.csv"
OUTPUT_PATH = AUDIT_DIR / "near_duplicate_candidates.csv"
SUMMARY_PATH = AUDIT_DIR / "near_duplicate_summary.json"

PHASH_THRESHOLD = 6
DHASH_THRESHOLD = 6


def hamming_distance(first_hash: int, second_hash: int) -> int:
    return bin(first_hash ^ second_hash).count("1")


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        records = list(csv.DictReader(file))

    if len(records) != 5558:
        raise ValueError(
            f"预期读取5558条记录，实际为{len(records)}条"
        )

    for record in records:
        record["phash_int"] = int(record["phash"], 16)
        record["dhash_int"] = int(record["dhash"], 16)

    candidates = []
    total_comparisons = 0
    cross_class_count = 0
    same_class_count = 0

    for first_index in range(len(records) - 1):
        first = records[first_index]

        for second_index in range(
            first_index + 1,
            len(records),
        ):
            second = records[second_index]
            total_comparisons += 1

            phash_distance = hamming_distance(
                first["phash_int"],
                second["phash_int"],
            )

            if phash_distance > PHASH_THRESHOLD:
                continue

            dhash_distance = hamming_distance(
                first["dhash_int"],
                second["dhash_int"],
            )

            if dhash_distance > DHASH_THRESHOLD:
                continue

            same_class = (
                first["class_name"]
                == second["class_name"]
            )

            if same_class:
                same_class_count += 1
            else:
                cross_class_count += 1

            candidates.append({
                "first_path": first["relative_path"],
                "first_class": first["class_name"],
                "second_path": second["relative_path"],
                "second_class": second["class_name"],
                "phash_distance": phash_distance,
                "dhash_distance": dhash_distance,
                "same_class": same_class,
            })

        if (
            (first_index + 1) % 500 == 0
            or first_index == len(records) - 2
        ):
            print(
                f"已完成 {first_index + 1}/"
                f"{len(records) - 1} 个基准图片"
            )

    candidates.sort(
        key=lambda row: (
            row["same_class"],
            row["phash_distance"]
            + row["dhash_distance"],
            row["phash_distance"],
            row["first_path"],
            row["second_path"],
        )
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "first_path",
            "first_class",
            "second_path",
            "second_class",
            "phash_distance",
            "dhash_distance",
            "same_class",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(candidates)

    summary = {
        "eligible_images": len(records),
        "total_pair_comparisons": total_comparisons,
        "phash_threshold": PHASH_THRESHOLD,
        "dhash_threshold": DHASH_THRESHOLD,
        "candidate_pair_count": len(candidates),
        "same_class_candidate_pairs": same_class_count,
        "cross_class_candidate_pairs": cross_class_count,
        "decision": (
            "Candidates require visual review; "
            "no automatic exclusion."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"总比较次数：{total_comparisons}")
    print(f"近重复候选对：{len(candidates)}")
    print(f"同类别候选对：{same_class_count}")
    print(f"跨类别候选对：{cross_class_count}")
    print(f"结果已保存：{OUTPUT_PATH}")
    print("本步骤没有排除、删除或修改图片")


if __name__ == "__main__":
    main()
