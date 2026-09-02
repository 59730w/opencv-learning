"""Build role-separated CRDLD manifests for the narrowed positive-geometry scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROLE_BY_SPLIT = {
    "train_data": "train_development",
    "validation_data": "validation_development",
    "test_data": "same_source_internal_benchmark",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**12, path.name)


def build_scoped_manifests(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: dict[str, str] = {}
    records_by_role: dict[str, list[dict[str, str]]] = {}
    excluded_cross_role: list[str] = []
    excluded_within_role: list[str] = []

    for split, role in ROLE_BY_SPLIT.items():
        image_dir = dataset_root / split / "image"
        label_dir = dataset_root / split / "label"
        images = {path.stem: path for path in image_dir.glob("*.jpg")}
        labels = {path.stem: path for path in label_dir.glob("*.jpg")}
        if images.keys() != labels.keys():
            raise ValueError(
                f"{split} pairing mismatch: missing_labels={sorted(images.keys() - labels.keys())}, "
                f"orphan_labels={sorted(labels.keys() - images.keys())}"
            )

        records: list[dict[str, str]] = []
        role_hashes: set[str] = set()
        for image_path in sorted(images.values(), key=_sort_key):
            item_key = f"{split}/{image_path.stem}"
            digest = _sha256(image_path)
            if digest in role_hashes:
                excluded_within_role.append(item_key)
                continue
            if digest in seen_hashes:
                excluded_cross_role.append(item_key)
                continue
            role_hashes.add(digest)
            seen_hashes[digest] = item_key
            label_path = labels[image_path.stem]
            records.append(
                {
                    "source_id": "crdld_v2_1",
                    "source_split": split,
                    "item_id": image_path.stem,
                    "role": role,
                    "image_path": image_path.relative_to(dataset_root).as_posix(),
                    "label_path": label_path.relative_to(dataset_root).as_posix(),
                    "image_sha256": digest,
                    "highest_group_key": "NOT_AVAILABLE",
                    "allowed_claim": "same_source_positive_geometry_only",
                }
            )
        records_by_role[role] = records
        manifest_path = output_dir / f"crdld_{role}_manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "same_source_positive_geometry_learning",
        "counts": {role: len(records) for role, records in records_by_role.items()},
        "excluded_cross_role_duplicates": excluded_cross_role,
        "excluded_within_role_duplicates": excluded_within_role,
        "scoped_gate": "PASS",
        "full_reject_aware_gate": "BLOCKED",
        "full_gate_blockers": [
            "crdld_dataset_license_not_explicit",
            "crdld_highest_leakage_groups_unavailable",
            "target_domain_reject_negatives_missing",
        ],
        "external_frozen_data_may_influence_development": False,
    }
    (output_dir / "scoped_crdld_manifest_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_scoped_manifests(args.dataset_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
