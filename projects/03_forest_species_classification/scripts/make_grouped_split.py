import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"

GROUPS_PATH = PROCESSED_DIR / "similarity_groups.csv"
MANIFEST_PATH = PROCESSED_DIR / "split_manifest.csv"
CLASS_MAP_PATH = PROCESSED_DIR / "class_to_idx.json"
SUMMARY_PATH = AUDIT_DIR / "split_summary.json"

SEED = 42
SPLITS = ("train", "validation", "test")
RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def assign_class_groups(class_name, groups):
    rng = random.Random(f"{SEED}:{class_name}")

    # 先随机打破相同大小组之间的顺序，再按组大小降序分配。
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total_images = sum(len(group) for group in groups)
    targets = {
        split: total_images * RATIOS[split]
        for split in SPLITS
    }
    assigned_counts = {split: 0 for split in SPLITS}
    assignments = {}

    if len(groups) < len(SPLITS):
        raise ValueError(
            f"类别 {class_name} 只有{len(groups)}个独立组，"
            "无法同时覆盖三个数据子集"
        )

    # 先保证每个子集至少得到一个组。
    for split, group in zip(SPLITS, groups[:3]):
        group_id = group[0]["group_id"]
        assignments[group_id] = split
        assigned_counts[split] += len(group)

    # 剩余组优先放入当前缺口最大的子集。
    for group in groups[3:]:
        deficits = {
            split: targets[split] - assigned_counts[split]
            for split in SPLITS
        }

        selected_split = max(
            SPLITS,
            key=lambda split: (
                deficits[split],
                -assigned_counts[split],
            ),
        )

        group_id = group[0]["group_id"]
        assignments[group_id] = selected_split
        assigned_counts[selected_split] += len(group)

    return assignments


def main():
    with GROUPS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        records = list(csv.DictReader(file))

    if len(records) != 5558:
        raise ValueError(
            f"预期5558条记录，实际为{len(records)}条"
        )

    class_names = sorted({
        row["class_name"] for row in records
    })

    if len(class_names) != 50:
        raise ValueError(
            f"预期50个类别，实际为{len(class_names)}个"
        )

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    records_by_class_and_group = defaultdict(
        lambda: defaultdict(list)
    )

    for row in records:
        records_by_class_and_group[
            row["class_name"]
        ][row["group_id"]].append(row)

    group_assignments = {}

    for class_name in class_names:
        groups = list(
            records_by_class_and_group[class_name].values()
        )

        assignments = assign_class_groups(
            class_name,
            groups,
        )

        for group_id, split in assignments.items():
            if group_id in group_assignments:
                raise ValueError(
                    f"group_id重复分配：{group_id}"
                )

            group_assignments[group_id] = split

    output_rows = []

    for row in sorted(
        records,
        key=lambda item: item["relative_path"],
    ):
        output_rows.append({
            "relative_path": row["relative_path"],
            "class_name": row["class_name"],
            "class_index": class_to_idx[row["class_name"]],
            "group_id": row["group_id"],
            "group_size": row["group_size"],
            "split": group_assignments[row["group_id"]],
        })

    # 验证路径唯一。
    paths = [
        row["relative_path"] for row in output_rows
    ]

    if len(paths) != len(set(paths)):
        raise ValueError("划分清单中存在重复图片路径")

    # 验证group_id没有跨数据子集。
    splits_by_group = defaultdict(set)

    for row in output_rows:
        splits_by_group[row["group_id"]].add(
            row["split"]
        )

    leaking_groups = {
        group_id: sorted(splits)
        for group_id, splits in splits_by_group.items()
        if len(splits) > 1
    }

    if leaking_groups:
        raise ValueError(
            f"发现跨子集group_id：{leaking_groups}"
        )

    # 验证每个类别都出现在三个子集中。
    splits_by_class = defaultdict(set)

    for row in output_rows:
        splits_by_class[row["class_name"]].add(
            row["split"]
        )

    incomplete_classes = {
        class_name: sorted(splits)
        for class_name, splits in splits_by_class.items()
        if splits != set(SPLITS)
    }

    if incomplete_classes:
        raise ValueError(
            f"以下类别没有覆盖三个子集：{incomplete_classes}"
        )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "class_name",
                "class_index",
                "group_id",
                "group_size",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    CLASS_MAP_PATH.write_text(
        json.dumps(
            class_to_idx,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    split_counts = Counter(
        row["split"] for row in output_rows
    )
    group_counts = Counter(group_assignments.values())

    per_class_split_counts = defaultdict(Counter)

    for row in output_rows:
        per_class_split_counts[row["class_name"]][
            row["split"]
        ] += 1

    summary = {
        "seed": SEED,
        "total_images": len(output_rows),
        "class_count": len(class_names),
        "total_groups": len(group_assignments),
        "target_ratios": RATIOS,
        "image_counts": dict(split_counts),
        "image_ratios": {
            split: split_counts[split] / len(output_rows)
            for split in SPLITS
        },
        "group_counts": dict(group_counts),
        "group_leakage_count": len(leaking_groups),
        "classes_missing_a_split": len(
            incomplete_classes
        ),
        "per_class_split_counts": {
            class_name: dict(
                per_class_split_counts[class_name]
            )
            for class_name in class_names
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"总图片数：{len(output_rows)}")
    print(f"类别数：{len(class_names)}")
    print(f"总组数：{len(group_assignments)}")
    print(f"图片划分：{dict(split_counts)}")
    print(f"分组划分：{dict(group_counts)}")
    print(f"跨子集group_id：{len(leaking_groups)}")
    print(
        f"未覆盖三个子集的类别："
        f"{len(incomplete_classes)}"
    )
    print(f"划分清单：{MANIFEST_PATH}")
    print("本步骤没有复制、移动或删除图片")


if __name__ == "__main__":
    main()