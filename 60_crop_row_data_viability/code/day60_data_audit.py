"""Day60 dataset-pair audit and evidence-role gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


REQUIRED_FIELDS = (
    "available",
    "license_verified",
    "task_match",
    "labels_support_metrics",
    "provenance_groups_known",
    "has_required_negatives",
)
REQUIRED_ROLES = ("id_development", "ood_development", "frozen_external_test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _duplicate_groups(paths: dict[str, Path]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for stem, path in paths.items():
        by_hash[_sha256(path)].append(stem)
    return sorted(sorted(group) for group in by_hash.values() if len(group) > 1)


def audit_image_mask_pairs(root: Path) -> dict[str, Any]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    images = {path.stem: path for path in image_dir.glob("*.jpg")}
    masks = {path.stem: path for path in mask_dir.glob("*.png")}

    decode_failures: list[str] = []
    shape_mismatches: list[str] = []
    shape_counts: Counter[str] = Counter()
    mask_value_counts: Counter[str] = Counter()

    for stem in sorted(images.keys() & masks.keys()):
        image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks[stem]), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            decode_failures.append(stem)
            continue
        image_shape = (image.shape[1], image.shape[0])
        mask_shape = (mask.shape[1], mask.shape[0])
        if image_shape != mask_shape:
            shape_mismatches.append(stem)
        shape_counts[f"{image_shape[0]}x{image_shape[1]}"] += 1
        values = ",".join(str(int(value)) for value in sorted(set(mask.reshape(-1))))
        mask_value_counts[values] += 1

    return {
        "image_count": len(images),
        "mask_count": len(masks),
        "missing_images": sorted(masks.keys() - images.keys()),
        "missing_masks": sorted(images.keys() - masks.keys()),
        "decode_failures": decode_failures,
        "shape_mismatches": shape_mismatches,
        "shape_counts": dict(sorted(shape_counts.items())),
        "mask_value_counts": dict(sorted(mask_value_counts.items())),
        "duplicate_image_groups": _duplicate_groups(images),
        "duplicate_mask_groups": _duplicate_groups(masks),
    }


def evaluate_data_gate(registry: dict[str, Any]) -> dict[str, Any]:
    sources = [
        source
        for source in registry.get("sources", [])
        if source.get("selected_for_gate", True)
    ]
    blockers: list[str] = []

    for source in sources:
        source_id = source.get("id", "unnamed")
        for field in REQUIRED_FIELDS:
            if not source.get(field, False):
                label = {
                    "available": "not_available",
                    "license_verified": "license_unverified",
                    "task_match": "task_mismatch",
                    "labels_support_metrics": "labels_do_not_support_metrics",
                    "provenance_groups_known": "provenance_groups_unknown",
                    "has_required_negatives": "required_negatives_missing",
                }[field]
                blockers.append(f"{source_id}:{label}")
        if source.get("role") == "frozen_external_test" and not source.get(
            "independent_of_id", False
        ):
            blockers.append(f"{source_id}:not_independent_of_id")

    for role in REQUIRED_ROLES:
        role_sources = [source for source in sources if source.get("role") == role]
        if not role_sources:
            blockers.append(f"missing_{role}")

    blockers = sorted(set(blockers))
    return {"status": "PASS" if not blockers else "BLOCKED", "blockers": blockers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {"pair_audit": audit_image_mask_pairs(args.dataset_root)}
    if args.registry:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        result["gate"] = evaluate_data_gate(registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
