import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

HASH_PATH = AUDIT_DIR / "perceptual_hashes.csv"
CANDIDATE_PATH = AUDIT_DIR / "near_duplicate_candidates.csv"
OUTPUT_PATH = PROCESSED_DIR / "similarity_groups.csv"
SUMMARY_PATH = AUDIT_DIR / "similarity_group_summary.json"


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root == second_root:
            return

        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root

        self.parent[second_root] = first_root

        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def main():
    with HASH_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        records = list(csv.DictReader(file))

    record_by_path = {
        row["relative_path"]: row
        for row in records
    }

    if len(record_by_path) != 5558:
        raise ValueError(
            f"预期5558张候选图片，实际为{len(record_by_path)}张"
        )

    union_find = UnionFind(record_by_path)

    with CANDIDATE_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        candidate_pairs = list(csv.DictReader(file))

    for pair in candidate_pairs:
        first_path = pair["first_path"]
        second_path = pair["second_path"]

        if first_path not in record_by_path:
            raise ValueError(f"未知路径：{first_path}")

        if second_path not in record_by_path:
            raise ValueError(f"未知路径：{second_path}")

        union_find.union(first_path, second_path)

    components = defaultdict(list)

    for relative_path in sorted(record_by_path):
        root = union_find.find(relative_path)
        components[root].append(relative_path)

    sorted_components = sorted(
        components.values(),
        key=lambda paths: (
            -len(paths),
            paths[0],
        ),
    )

    output_rows = []
    non_singleton_group_count = 0
    cross_class_group_count = 0

    for group_index, paths in enumerate(
        sorted_components,
        start=1,
    ):
        group_id = f"group_{group_index:04d}"
        group_size = len(paths)

        class_names = sorted({
            record_by_path[path]["class_name"]
            for path in paths
        })

        if group_size > 1:
            non_singleton_group_count += 1

        if len(class_names) > 1:
            cross_class_group_count += 1

        for relative_path in paths:
            record = record_by_path[relative_path]

            output_rows.append({
                "relative_path": relative_path,
                "class_name": record["class_name"],
                "group_id": group_id,
                "group_size": group_size,
                "is_similarity_group": group_size > 1,
            })

    if len(output_rows) != 5558:
        raise ValueError(
            f"输出记录应为5558条，实际为{len(output_rows)}条"
        )

    group_size_counts = Counter(
        len(paths) for paths in sorted_components
    )

    summary = {
        "eligible_images": len(record_by_path),
        "candidate_pair_count": len(candidate_pairs),
        "total_groups_including_singletons": len(sorted_components),
        "non_singleton_similarity_groups": non_singleton_group_count,
        "cross_class_similarity_groups": cross_class_group_count,
        "largest_group_size": max(
            len(paths) for paths in sorted_components
        ),
        "group_size_distribution": {
            str(size): count
            for size, count in sorted(group_size_counts.items())
        },
        "split_rule": (
            "All images with the same group_id must be assigned "
            "to the same train, validation or test split."
        ),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
                "group_id",
                "group_size",
                "is_similarity_group",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"候选图片：{len(record_by_path)}")
    print(f"候选关系：{len(candidate_pairs)}")
    print(f"总分组数：{len(sorted_components)}")
    print(f"非单张相似组：{non_singleton_group_count}")
    print(f"最大组图片数：{summary['largest_group_size']}")
    print(f"跨类别相似组：{cross_class_group_count}")
    print(f"分组清单：{OUTPUT_PATH}")
    print("原始图片未删除或修改")


if __name__ == "__main__":
    main()