"""Day60 dataset-pair audit and evidence-role gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def _horizontal_run_centers(mask: np.ndarray, y_norm: float) -> list[float]:
    height = mask.shape[0]
    y = round(y_norm * (height - 1))
    half_height = max(1, round(0.006 * height))
    band = mask[max(0, y - half_height) : min(height, y + half_height + 1)]
    active = np.any(band >= 128, axis=0)
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    denominator = max(1, mask.shape[1] - 1)
    return [float((start + end - 1) / 2.0 / denominator) for start, end in zip(starts, ends)]


def _count_horizontal_runs(mask: np.ndarray, y_norm: float) -> int:
    return len(_horizontal_run_centers(mask, y_norm))


def audit_multirow_label_masks(label_dir: Path) -> dict[str, Any]:
    """Audit whether merged centerline masks contain separable multi-row signal.

    The result deliberately does not claim instance identities: CRDLD stores all
    row centerlines in one binary JPEG mask.  Counts at fixed horizontal bands
    establish only that multiple row crossings are present and derivable.
    """
    if not label_dir.is_dir():
        raise ValueError("label directory does not exist")
    paths = sorted(label_dir.glob("*.jpg"))
    audit_bands = (0.25, 0.40, 0.60, 0.80)
    decode_failures: list[str] = []
    per_image_max: list[int] = []
    centered_crop_row_count = 0
    both_sides_count = 0
    corridor_proxy_clear_count = 0
    band_counts: dict[str, Counter[int]] = {
        f"{band:.2f}": Counter() for band in audit_bands
    }
    for path in paths:
        label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            decode_failures.append(path.stem)
            continue
        counts = [_count_horizontal_runs(label, band) for band in audit_bands]
        corridor_centers = _horizontal_run_centers(label, 0.80)
        centered = any(abs(value - 0.5) <= 0.04 for value in corridor_centers)
        both_sides = any(value < 0.46 for value in corridor_centers) and any(
            value > 0.54 for value in corridor_centers
        )
        centered_crop_row_count += int(centered)
        both_sides_count += int(both_sides)
        corridor_proxy_clear_count += int(both_sides and not centered)
        per_image_max.append(max(counts))
        for band, count in zip(audit_bands, counts):
            band_counts[f"{band:.2f}"][count] += 1

    decoded = len(per_image_max)
    multirow = sum(count >= 2 for count in per_image_max)
    return {
        "label_count": len(paths),
        "decoded_count": decoded,
        "decode_failures": decode_failures,
        "audit_y_norms": list(audit_bands),
        "instance_ids_available": False,
        "label_representation": "merged_binary_centerline_mask",
        "max_rows_at_any_audit_band": max(per_image_max, default=0),
        "multirow_signal_present_fraction": multirow / decoded if decoded else 0.0,
        "corridor_audit_y_norm": 0.80,
        "camera_center_exclusion_norm": 0.04,
        "centered_crop_row_at_corridor_band_fraction": centered_crop_row_count / decoded if decoded else 0.0,
        "both_sides_at_corridor_band_fraction": both_sides_count / decoded if decoded else 0.0,
        "corridor_proxy_clear_fraction": corridor_proxy_clear_count / decoded if decoded else 0.0,
        "row_count_distributions_by_band": {
            band: dict(sorted(counts.items())) for band, counts in band_counts.items()
        },
        "formal_instance_metrics_require_derived_ordered_matching": True,
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
