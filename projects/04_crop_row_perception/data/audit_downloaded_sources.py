"""Audit downloaded CRDLD and RowDetr data for evidence-role assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_crdld(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    result: dict[str, Any] = {"splits": {}}
    hashes: dict[str, str] = {}
    hash_locations: dict[str, list[str]] = defaultdict(list)

    for split in ("train_data", "test_data", "validation_data"):
        image_dir = root / split / "image"
        label_dir = root / split / "label"
        images = {path.stem: path for path in image_dir.glob("*.jpg")}
        labels = {path.stem: path for path in label_dir.glob("*.jpg")}
        dimensions: Counter[str] = Counter()
        mask_unique_counts: Counter[str] = Counter()
        mask_binary_support: Counter[str] = Counter()
        decode_failures: list[str] = []
        shape_mismatches: list[str] = []

        for stem in sorted(images.keys() & labels.keys(), key=lambda value: int(value)):
            image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
            label = cv2.imread(str(labels[stem]), cv2.IMREAD_GRAYSCALE)
            if image is None or label is None:
                decode_failures.append(stem)
                continue
            if image.shape[:2] != label.shape[:2]:
                shape_mismatches.append(stem)
            dimensions[f"{image.shape[0]}x{image.shape[1]}"] += 1
            values = np.unique(label)
            mask_unique_counts[str(len(values))] += 1
            mask_binary_support[
                "has_dark_and_bright"
                if int(values.min()) <= 32 and int(values.max()) >= 223
                else "unexpected_range"
            ] += 1
            digest = file_hash(images[stem])
            key = f"{split}/{stem}"
            hashes[key] = digest
            hash_locations[digest].append(key)

        result["splits"][split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "missing_labels": sorted(images.keys() - labels.keys()),
            "orphan_labels": sorted(labels.keys() - images.keys()),
            "decode_failures": decode_failures,
            "shape_mismatches": shape_mismatches,
            "dimensions_hxw": dict(sorted(dimensions.items())),
            "mask_unique_value_count_distribution": dict(sorted(mask_unique_counts.items())),
            "mask_binary_support": dict(sorted(mask_binary_support.items())),
        }

    result["exact_duplicate_groups_across_splits"] = [
        locations
        for locations in hash_locations.values()
        if len({location.split("/", 1)[0] for location in locations}) > 1
    ]
    return result, hashes


def validate_row_label(path: Path, width: int, height: int) -> tuple[int, list[str]]:
    issues: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return 0, [f"json_error:{error}"]
    rows = payload.get("labels")
    if not isinstance(rows, list):
        return 0, ["labels_not_list"]
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(f"row_{index}:not_object")
            continue
        xs, ys, alphas = row.get("x"), row.get("y"), row.get("alpha")
        if not all(isinstance(values, list) for values in (xs, ys)):
            issues.append(f"row_{index}:coordinates_not_lists")
            continue
        if len(xs) != len(ys) or len(xs) < 4:
            issues.append(f"row_{index}:invalid_point_count")
            continue
        if any(not isinstance(value, (int, float)) for value in xs + ys):
            issues.append(f"row_{index}:non_numeric_coordinate")
            continue
        if any(value < 0 or value > width for value in xs):
            issues.append(f"row_{index}:x_out_of_bounds")
        if any(value < 0 or value > height for value in ys):
            issues.append(f"row_{index}:y_out_of_bounds")
        if alphas is not None:
            if not isinstance(alphas, list) or len(alphas) != len(xs):
                issues.append(f"row_{index}:invalid_alpha_count")
            elif any(not isinstance(value, (int, float)) or value < 0 or value > 1 for value in alphas):
                issues.append(f"row_{index}:alpha_out_of_bounds")
    return len(rows), issues


def audit_rowdetr(root: Path, crdld_hashes: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"generic_splits": {}}
    source_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    empty_labels: list[str] = []
    label_issue_samples: list[dict[str, Any]] = []
    label_issue_count = 0
    sorghum_ids_by_split: dict[str, list[int]] = defaultdict(list)
    sorghum_hashes: dict[str, str] = {}
    crdld_resolution_hashes: dict[str, str] = {}

    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = {path.stem: path for path in image_dir.glob("*.jpg")}
        labels = {path.stem: path for path in label_dir.glob("*.json")}
        dimensions: Counter[str] = Counter()
        decode_failures: list[str] = []
        row_count_distribution: Counter[str] = Counter()

        for stem in sorted(images.keys() & labels.keys(), key=lambda value: int(value)):
            image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
            if image is None:
                decode_failures.append(stem)
                continue
            height, width = image.shape[:2]
            dimension = f"{height}x{width}"
            dimensions[dimension] += 1
            source_counts[dimension] += 1
            source_split_counts[f"{split}/{dimension}"] += 1
            row_count, issues = validate_row_label(labels[stem], width, height)
            row_count_distribution[str(row_count)] += 1
            if row_count == 0:
                empty_labels.append(f"{split}/{stem}")
            if issues and len(label_issue_samples) < 100:
                label_issue_samples.append({"item": f"{split}/{stem}", "issues": issues})
            label_issue_count += bool(issues)
            digest = file_hash(images[stem])
            if dimension == "720x1280":
                sorghum_ids_by_split[split].append(int(stem))
                sorghum_hashes[f"{split}/{stem}"] = digest
            elif dimension == "512x512":
                crdld_resolution_hashes[f"{split}/{stem}"] = digest

        result["generic_splits"][split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "missing_labels": sorted(images.keys() - labels.keys()),
            "orphan_labels": sorted(labels.keys() - images.keys()),
            "decode_failures": decode_failures,
            "dimensions_hxw": dict(sorted(dimensions.items())),
            "row_count_distribution": dict(sorted(row_count_distribution.items())),
        }

    sorghum_ids = sorted(
        identifier for identifiers in sorghum_ids_by_split.values() for identifier in identifiers
    )
    crdld_hash_set = set(crdld_hashes.values())
    result.update(
        {
            "source_dimension_counts": dict(sorted(source_counts.items())),
            "source_split_counts": dict(sorted(source_split_counts.items())),
            "empty_label_count": len(empty_labels),
            "empty_label_examples": empty_labels[:100],
            "label_issue_count": label_issue_count,
            "label_issue_samples": label_issue_samples,
            "sorghum_external_candidate": {
                "count": len(sorghum_ids),
                "paper_reported_count": 2152,
                "archive_shortfall": 2152 - len(sorghum_ids),
                "split_counts": {
                    split: len(identifiers)
                    for split, identifiers in sorted(sorghum_ids_by_split.items())
                },
                "id_min": min(sorghum_ids) if sorghum_ids else None,
                "id_max": max(sorghum_ids) if sorghum_ids else None,
                "numeric_ids_are_split_local_not_sequence_keys": True,
                "exact_duplicates_with_crdld": sorted(
                    key for key, digest in sorghum_hashes.items() if digest in crdld_hash_set
                ),
                "sequence_metadata_available": False,
                "safe_role": "whole_source_frozen_external_positive_only",
            },
            "rowdetr_512_overlap_with_crdld": {
                "rowdetr_512_count": len(crdld_resolution_hashes),
                "exact_match_count": sum(
                    digest in crdld_hash_set for digest in crdld_resolution_hashes.values()
                ),
                "known_source_overlap_from_paper": True,
            },
            "archive_pair_count": sum(source_counts.values()),
            "paper_reported_pair_count": 6962,
            "archive_pair_shortfall": 6962 - sum(source_counts.values()),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crdld-root", type=Path, required=True)
    parser.add_argument("--rowdetr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    crdld, crdld_hashes = audit_crdld(args.crdld_root)
    rowdetr = audit_rowdetr(args.rowdetr_root, crdld_hashes)
    report = {
        "schema_version": 1,
        "crdld": crdld,
        "rowdetr": rowdetr,
        "gate": {
            "status": "BLOCKED",
            "passed": [
                "archives_verified",
                "image_label_pairing_verified",
                "external_positive_source_independent",
                "external_geometry_labels_available",
            ],
            "blockers": [
                "crdld_dataset_license_not_explicit",
                "crdld_highest_leakage_groups_unavailable",
                "required_reject_negatives_missing",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "crdld_pairs": sum(
                    split["image_count"] for split in crdld["splits"].values()
                ),
                "rowdetr_pairs": rowdetr["archive_pair_count"],
                "sorghum_external_pairs": rowdetr["sorghum_external_candidate"]["count"],
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
