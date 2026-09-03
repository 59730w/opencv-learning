"""Day62 morphology and connected-region analysis for crop-row candidates.

The input candidate is the frozen Day61 bounded Gray-World plus HSV mask.
CRDLD labels are crop-row centerlines rather than vegetation masks, so all
reported line-neighborhood values are proxy metrics, not segmentation IoU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


DAY61_CODE_DIR = Path(__file__).resolve().parents[2] / "61_crop_row_color_illumination" / "code"
if str(DAY61_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DAY61_CODE_DIR))

from day61_color_illumination import (  # noqa: E402
    compute_line_proxy_metrics,
    proxy_robustness_score,
    read_manifest_stems,
    segment_grayworld_hsv,
)


RAW_CONFIG: dict[str, Any] = {
    "name": "raw_day61",
    "order": None,
    "open_kernel": 3,
    "close_kernel": 5,
    "min_area_fraction": 0.0,
}

MORPHOLOGY_CONFIGS: list[dict[str, Any]] = [
    {"name": "open3", "order": "open", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.0},
    {"name": "close5", "order": "close", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.0},
    {"name": "open3_close3", "order": "open_close", "open_kernel": 3, "close_kernel": 3, "min_area_fraction": 0.0},
    {"name": "open3_close5", "order": "open_close", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.0},
    {"name": "close5_open3", "order": "close_open", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.0},
    {"name": "open3_close5_area_00005", "order": "open_close", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.00005},
    {"name": "open3_close5_area_00010", "order": "open_close", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.00010},
    {"name": "open3_close5_area_00020", "order": "open_close", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.00020},
    {"name": "close5_open3_area_00010", "order": "close_open", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.00010},
    {"name": "close5_open3_area_00020", "order": "close_open", "open_kernel": 3, "close_kernel": 5, "min_area_fraction": 0.00020},
]

V1_FROZEN_CONFIG = next(
    config for config in MORPHOLOGY_CONFIGS if config["name"] == "open3_close5_area_00020"
)

V2_PERSPECTIVE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": f"perspective_top{int(top_scale * 100):02d}_exp{exponent}",
        "order": "open_close",
        "open_kernel": 3,
        "close_kernel": 5,
        "min_area_fraction": 0.00020,
        "perspective_top_scale": top_scale,
        "perspective_exponent": float(exponent),
    }
    for top_scale in (0.25, 0.40, 0.55, 0.70)
    for exponent in (1, 2)
]

V2_DIRECTIONAL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": f"directional_close{width}x{height}_fixed",
        "order": "open_close",
        "open_kernel": 3,
        "close_kernel": [width, height],
        "min_area_fraction": 0.00020,
    }
    for width, height in ((3, 7), (3, 9), (5, 7))
] + [
    {
        "name": f"directional_close{width}x{height}_perspective40",
        "order": "open_close",
        "open_kernel": 3,
        "close_kernel": [width, height],
        "min_area_fraction": 0.00020,
        "perspective_top_scale": 0.40,
        "perspective_exponent": 1.0,
    }
    for width, height in ((3, 7), (3, 9), (5, 7))
]


def validate_binary_mask(mask: np.ndarray) -> None:
    """Require a two-dimensional uint8 mask containing only 0 and 255."""
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("expected a two-dimensional uint8 mask")
    if not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("binary mask values must be 0 or 255")


def _ellipse_kernel(size: int | tuple[int, int] | list[int]) -> np.ndarray:
    dimensions = (size, size) if isinstance(size, int) else tuple(size)
    if (
        len(dimensions) != 2
        or any(value <= 0 or value % 2 == 0 for value in dimensions)
    ):
        raise ValueError("kernel dimensions must be positive odd integers")
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dimensions)


def apply_morphology(
    mask: np.ndarray,
    open_kernel: int | tuple[int, int] | list[int] = 3,
    close_kernel: int | tuple[int, int] | list[int] = 5,
    order: str = "open_close",
) -> np.ndarray:
    """Apply deterministic opening/closing while preserving mask invariants."""
    validate_binary_mask(mask)
    opening_kernel = _ellipse_kernel(open_kernel)
    closing_kernel = _ellipse_kernel(close_kernel)
    operations = {
        "open": (cv2.MORPH_OPEN, opening_kernel),
        "close": (cv2.MORPH_CLOSE, closing_kernel),
    }
    sequences = {
        "open": ("open",),
        "close": ("close",),
        "open_close": ("open", "close"),
        "close_open": ("close", "open"),
    }
    if order not in sequences:
        raise ValueError(f"unsupported morphology order: {order}")
    output = mask.copy()
    for name in sequences[order]:
        operation, kernel = operations[name]
        output = cv2.morphologyEx(output, operation, kernel)
    validate_binary_mask(output)
    return output


def component_records(mask: np.ndarray) -> list[dict[str, Any]]:
    """Return foreground connected-component statistics, largest first."""
    validate_binary_mask(mask)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    records: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        records.append(
            {
                "label": int(label),
                "area": int(area),
                "bbox": [int(x), int(y), int(width), int(height)],
                "centroid": [float(centroids[label, 0]), float(centroids[label, 1])],
            }
        )
    return sorted(records, key=lambda record: (-record["area"], record["label"]))


def contour_records(mask: np.ndarray) -> list[dict[str, Any]]:
    """Return external contour geometry for inspection and later shape filtering."""
    validate_binary_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    records: list[dict[str, Any]] = []
    for index, contour in enumerate(contours):
        x, y, width, height = cv2.boundingRect(contour)
        records.append(
            {
                "contour": int(index),
                "area": float(cv2.contourArea(contour)),
                "perimeter": float(cv2.arcLength(contour, True)),
                "bbox": [int(x), int(y), int(width), int(height)],
            }
        )
    return sorted(records, key=lambda record: (-record["area"], record["contour"]))


def filter_components_by_area(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected foreground regions smaller than ``min_area`` pixels."""
    validate_binary_mask(mask)
    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == label] = 255
    return output


def filter_components_perspective(
    mask: np.ndarray,
    base_area_fraction: float,
    top_scale: float = 0.4,
    exponent: float = 1.0,
) -> np.ndarray:
    """Use a smaller area threshold near the image top where plants appear smaller."""
    validate_binary_mask(mask)
    if base_area_fraction < 0:
        raise ValueError("base_area_fraction must be non-negative")
    if not 0 < top_scale <= 1:
        raise ValueError("top_scale must be in (0, 1]")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    output = np.zeros_like(mask)
    denominator = max(1, mask.shape[0] - 1)
    base_area = mask.size * base_area_fraction
    for label in range(1, count):
        center_y = float(centroids[label, 1]) / denominator
        scale = top_scale + (1.0 - top_scale) * center_y**exponent
        min_area = max(1, math.ceil(base_area * scale))
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == label] = 255
    return output


def vertical_line_support_metrics(
    candidate_mask: np.ndarray,
    line_mask: np.ndarray,
    bands: int = 8,
    dilation_radius: int = 7,
    support_threshold: float = 0.15,
) -> dict[str, float]:
    """Measure row-neighborhood support and continuity over horizontal bands."""
    validate_binary_mask(candidate_mask)
    validate_binary_mask(line_mask)
    if candidate_mask.shape != line_mask.shape:
        raise ValueError("candidate and line masks must have the same shape")
    if bands <= 0:
        raise ValueError("bands must be positive")
    kernel_size = 2 * dilation_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    neighborhood = cv2.dilate((line_mask > 0).astype(np.uint8), kernel) > 0
    candidate = candidate_mask > 0
    band_supports: list[float] = []
    for indices in np.array_split(np.arange(candidate.shape[0]), bands):
        band_neighborhood = neighborhood[indices]
        if band_neighborhood.any():
            band_supports.append(float(candidate[indices][band_neighborhood].mean()))
    if not band_supports:
        return {
            "mean_band_support": 0.0,
            "p10_band_support": 0.0,
            "supported_band_fraction": 0.0,
            "longest_supported_run_fraction": 0.0,
        }
    supported = [value >= support_threshold for value in band_supports]
    longest = current = 0
    for value in supported:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "mean_band_support": float(np.mean(band_supports)),
        "p10_band_support": float(np.percentile(band_supports, 10)),
        "supported_band_fraction": float(np.mean(supported)),
        "longest_supported_run_fraction": float(longest / len(supported)),
    }


def region_change_metrics(
    raw_mask: np.ndarray, cleaned_mask: np.ndarray, small_area: int = 32
) -> dict[str, float | int]:
    """Describe fragmentation and pixel changes without segmentation claims."""
    validate_binary_mask(raw_mask)
    validate_binary_mask(cleaned_mask)
    if raw_mask.shape != cleaned_mask.shape:
        raise ValueError("raw and cleaned masks must have the same shape")
    raw_records = component_records(raw_mask)
    cleaned_records = component_records(cleaned_mask)
    raw_foreground = raw_mask > 0
    cleaned_foreground = cleaned_mask > 0
    raw_count = len(raw_records)
    return {
        "raw_component_count": raw_count,
        "cleaned_component_count": len(cleaned_records),
        "component_reduction_fraction": (
            float((raw_count - len(cleaned_records)) / raw_count) if raw_count else 0.0
        ),
        "raw_small_component_count": sum(
            record["area"] < small_area for record in raw_records
        ),
        "cleaned_small_component_count": sum(
            record["area"] < small_area for record in cleaned_records
        ),
        "removed_raw_foreground_fraction": (
            float((raw_foreground & ~cleaned_foreground).sum() / raw_foreground.sum())
            if raw_foreground.any()
            else 0.0
        ),
        "added_image_fraction": float((~raw_foreground & cleaned_foreground).mean()),
    }


def split_development_stems(stems: list[str]) -> dict[str, list[str]]:
    """Split 1,250 CRDLD development IDs into search/review/stress roles."""
    if len(stems) != 1250 or len(set(stems)) != 1250:
        raise ValueError("expected 1,250 unique train-development item IDs")
    ranked = sorted(
        stems,
        key=lambda stem: (
            hashlib.sha256(f"day62-morphology:{stem}".encode()).hexdigest(),
            int(stem),
        ),
    )
    return {
        "search": ranked[:256],
        "review": ranked[256:512],
        "stress": ranked[512:],
    }


def split_cv_folds(stems: list[str], folds: int = 5) -> list[list[str]]:
    """Create deterministic balanced folds for stable Day62 v2 selection."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("stems must be non-empty and unique")
    ranked = sorted(
        stems,
        key=lambda stem: (
            hashlib.sha256(f"day62-v2-fold:{stem}".encode()).hexdigest(),
            int(stem),
        ),
    )
    return [ranked[index::folds] for index in range(folds)]


def clean_candidate_mask(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Apply one declared morphology/area configuration to a Day61 mask."""
    validate_binary_mask(mask)
    order = config.get("order")
    cleaned = (
        mask.copy()
        if order is None
        else apply_morphology(
            mask,
            open_kernel=config["open_kernel"],
            close_kernel=config["close_kernel"],
            order=str(order),
        )
    )
    fraction = float(config.get("min_area_fraction", 0.0))
    if fraction < 0:
        raise ValueError("min_area_fraction must be non-negative")
    if fraction > 0:
        if "perspective_top_scale" in config:
            cleaned = filter_components_perspective(
                cleaned,
                base_area_fraction=fraction,
                top_scale=float(config["perspective_top_scale"]),
                exponent=float(config.get("perspective_exponent", 1.0)),
            )
        else:
            cleaned = filter_components_by_area(
                cleaned, min_area=max(1, math.ceil(cleaned.size * fraction))
            )
    validate_binary_mask(cleaned)
    return cleaned


def acceptance_checks(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, bool]:
    """Pre-registered gate for useful cleanup without proxy-metric collapse."""
    return {
        "mean_support_drop_at_most_0_02": (
            candidate["mean_support"] >= baseline["mean_support"] - 0.02
        ),
        "mean_gap_drop_at_most_0_005": (
            candidate["mean_gap"] >= baseline["mean_gap"] - 0.005
        ),
        "p10_gap_drop_at_most_0_01": (
            candidate["p10_gap"] >= baseline["p10_gap"] - 0.01
        ),
        "max_off_line_increase_at_most_0_02": (
            candidate["max_off_line_activation"]
            <= baseline["max_off_line_activation"] + 0.02
        ),
        "mean_component_reduction_at_least_0_20": (
            candidate["mean_component_reduction_fraction"] >= 0.20
        ),
        "mean_removed_raw_foreground_at_most_0_15": (
            candidate["mean_removed_raw_foreground_fraction"] <= 0.15
        ),
        "mean_added_image_fraction_at_most_0_03": (
            candidate["mean_added_image_fraction"] <= 0.03
        ),
    }


def v2_acceptance_checks(
    v1: dict[str, float], candidate: dict[str, float]
) -> dict[str, bool]:
    """Require perspective-aware v2 to improve preservation without losing v1 quality."""
    return {
        "mean_support_drop_at_most_0_005": (
            candidate["mean_support"] >= v1["mean_support"] - 0.005
        ),
        "mean_gap_drop_at_most_0_003": (
            candidate["mean_gap"] >= v1["mean_gap"] - 0.003
        ),
        "p10_gap_drop_at_most_0_005": (
            candidate["p10_gap"] >= v1["p10_gap"] - 0.005
        ),
        "max_off_line_increase_at_most_0_01": (
            candidate["max_off_line_activation"]
            <= v1["max_off_line_activation"] + 0.01
        ),
        "mean_component_reduction_at_least_0_75": (
            candidate["mean_component_reduction_fraction"] >= 0.75
        ),
        "mean_component_count_at_most_1_5x_v1": (
            candidate["mean_cleaned_component_count"]
            <= 1.5 * max(v1["mean_cleaned_component_count"], 1.0)
        ),
        "removed_foreground_improves_at_least_0_002": (
            candidate["mean_removed_raw_foreground_fraction"]
            <= v1["mean_removed_raw_foreground_fraction"] - 0.002
        ),
        "mean_added_image_fraction_at_most_0_03": (
            candidate["mean_added_image_fraction"] <= 0.03
        ),
        "mean_vertical_support_drop_at_most_0_002": (
            candidate["mean_vertical_band_support"]
            >= v1["mean_vertical_band_support"] - 0.002
        ),
        "p10_vertical_support_drop_at_most_0_01": (
            candidate["p10_image_vertical_band_support"]
            >= v1["p10_image_vertical_band_support"] - 0.01
        ),
        "supported_band_fraction_drop_at_most_0_02": (
            candidate["mean_supported_band_fraction"]
            >= v1["mean_supported_band_fraction"] - 0.02
        ),
        "longest_run_drop_at_most_0_02": (
            candidate["mean_longest_supported_run_fraction"]
            >= v1["mean_longest_supported_run_fraction"] - 0.02
        ),
        "geometry_or_preservation_gain": (
            candidate["mean_vertical_band_support"]
            >= v1["mean_vertical_band_support"] + 0.002
            or candidate["p10_image_vertical_band_support"]
            >= v1["p10_image_vertical_band_support"] + 0.005
            or candidate["mean_removed_raw_foreground_fraction"]
            <= v1["mean_removed_raw_foreground_fraction"] - 0.005
        ),
    }


def select_v2_from_fold_summaries(
    fold_summaries: list[dict[str, dict[str, Any]]],
    baseline_name: str,
    candidate_names: list[str],
) -> tuple[str, dict[str, int]]:
    """Select a v2 candidate only when it passes at least 80% of folds."""
    if not fold_summaries or not candidate_names:
        raise ValueError("fold summaries and candidate names must be non-empty")
    required_passes = math.ceil(0.8 * len(fold_summaries))
    pass_counts: dict[str, int] = {}
    for name in candidate_names:
        pass_counts[name] = sum(
            all(v2_acceptance_checks(fold[baseline_name], fold[name]).values())
            for fold in fold_summaries
        )
    eligible = [name for name in candidate_names if pass_counts[name] >= required_passes]
    if not eligible:
        return baseline_name, pass_counts

    def utility(name: str) -> float:
        values = []
        for fold in fold_summaries:
            candidate = fold[name]
            values.append(
                float(candidate["score"])
                + 0.03 * float(candidate["mean_vertical_band_support"])
                + 0.02 * float(candidate["p10_image_vertical_band_support"])
                + 0.01 * float(candidate["mean_longest_supported_run_fraction"])
                - 0.05 * float(candidate["mean_removed_raw_foreground_fraction"])
            )
        return float(np.mean(values))

    return max(eligible, key=utility), pass_counts


def choose_candidate(
    baseline: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Choose the best eligible cleanup, or retain the frozen raw baseline."""
    eligible: list[tuple[dict[str, Any], dict[str, bool]]] = []
    for candidate in candidates:
        checks = acceptance_checks(baseline, candidate)
        if all(checks.values()):
            eligible.append((candidate, checks))
    if not eligible:
        return baseline, {"no_cleanup_candidate_passed": False}

    def utility(item: tuple[dict[str, Any], dict[str, bool]]) -> float:
        candidate = item[0]
        return (
            float(candidate["score"])
            + 0.05 * float(candidate["mean_component_reduction_fraction"])
            - 0.10 * float(candidate["mean_removed_raw_foreground_fraction"])
            - 0.10 * float(candidate["mean_added_image_fraction"])
        )

    return max(eligible, key=utility)


def _read_pair(
    image_dir: Path, label_dir: Path, stem: str
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    line = cv2.imread(str(label_dir / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    if image is None or line is None or image.shape[:2] != line.shape:
        raise ValueError(f"invalid image/label pair: {stem}")
    return image, np.where(line >= 128, 255, 0).astype(np.uint8)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize empty evaluation rows")
    proxy = proxy_robustness_score(
        [float(row["line_neighborhood_support"]) for row in rows],
        [float(row["off_line_activation"]) for row in rows],
    )
    proxy["mean_candidate_fraction"] = float(
        np.mean([row["candidate_fraction"] for row in rows])
    )
    for key in (
        "raw_component_count",
        "cleaned_component_count",
        "component_reduction_fraction",
        "raw_small_component_count",
        "cleaned_small_component_count",
        "removed_raw_foreground_fraction",
        "added_image_fraction",
    ):
        proxy[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
    proxy["median_raw_component_count"] = float(
        np.median([row["raw_component_count"] for row in rows])
    )
    proxy["median_cleaned_component_count"] = float(
        np.median([row["cleaned_component_count"] for row in rows])
    )
    proxy["mean_vertical_band_support"] = float(
        np.mean([row["mean_band_support"] for row in rows])
    )
    proxy["p10_image_vertical_band_support"] = float(
        np.percentile([row["mean_band_support"] for row in rows], 10)
    )
    proxy["mean_supported_band_fraction"] = float(
        np.mean([row["supported_band_fraction"] for row in rows])
    )
    proxy["p10_supported_band_fraction"] = float(
        np.percentile([row["supported_band_fraction"] for row in rows], 10)
    )
    proxy["mean_longest_supported_run_fraction"] = float(
        np.mean([row["longest_supported_run_fraction"] for row in rows])
    )
    proxy["p10_longest_supported_run_fraction"] = float(
        np.percentile([row["longest_supported_run_fraction"] for row in rows], 10)
    )
    return proxy


def evaluate_configurations(
    image_dir: Path,
    label_dir: Path,
    stems: list[str],
    configs: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Evaluate morphology configurations on declared positive-development IDs."""
    if not stems or not configs:
        raise ValueError("stems and configs must be non-empty")
    rows_by_config: dict[str, list[dict[str, Any]]] = {
        str(config["name"]): [] for config in configs
    }
    for stem in stems:
        image, line = _read_pair(image_dir, label_dir, stem)
        raw_mask = segment_grayworld_hsv(image)
        small_area = max(1, math.ceil(raw_mask.size * 0.0001))
        for config in configs:
            name = str(config["name"])
            cleaned = clean_candidate_mask(raw_mask, config)
            proxy = compute_line_proxy_metrics(cleaned, line)
            vertical = vertical_line_support_metrics(cleaned, line)
            region = region_change_metrics(raw_mask, cleaned, small_area=small_area)
            rows_by_config[name].append(
                {"stem": stem, "configuration": name, **proxy, **vertical, **region}
            )
    summaries = {
        name: {"name": name, **_summarize_rows(rows)}
        for name, rows in rows_by_config.items()
    }
    all_rows = [row for rows in rows_by_config.values() for row in rows]
    return summaries, all_rows


def _candidate_utility(candidate: dict[str, Any]) -> float:
    return (
        float(candidate["score"])
        + 0.05 * float(candidate["mean_component_reduction_fraction"])
        - 0.10 * float(candidate["mean_removed_raw_foreground_fraction"])
        - 0.10 * float(candidate["mean_added_image_fraction"])
    )


def _eligible_ranked(
    baseline: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    eligible = [
        candidate
        for candidate in candidates
        if all(acceptance_checks(baseline, candidate).values())
    ]
    return sorted(eligible, key=_candidate_utility, reverse=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _caption(tile: np.ndarray, text: str) -> np.ndarray:
    output = tile.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return output


def _color_tile(image: np.ndarray, title: str, size: int = 220) -> np.ndarray:
    return _caption(cv2.resize(image, (size, size)), title)


def _gray_tile(mask: np.ndarray, title: str, size: int = 220) -> np.ndarray:
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return _caption(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR), title)


def _region_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 255), 1)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 80, 0), 1)
    return output


def _select_review_stems(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    count: int,
) -> list[str]:
    baseline = {row["stem"]: row for row in baseline_rows}
    candidate = {row["stem"]: row for row in candidate_rows}
    stems = sorted(set(baseline) & set(candidate), key=int)
    best_cleanup = sorted(
        stems,
        key=lambda stem: candidate[stem]["component_reduction_fraction"],
        reverse=True,
    )
    worst_gap_change = sorted(
        stems,
        key=lambda stem: (
            candidate[stem]["line_neighborhood_support"]
            - candidate[stem]["off_line_activation"]
        )
        - (
            baseline[stem]["line_neighborhood_support"]
            - baseline[stem]["off_line_activation"]
        ),
    )
    selected: list[str] = []
    for stem in best_cleanup[: max(1, count // 2)] + worst_gap_change:
        if stem not in selected:
            selected.append(stem)
        if len(selected) == count:
            break
    return selected


def _write_contact_sheet(
    path: Path,
    image_dir: Path,
    label_dir: Path,
    stems: list[str],
    frozen_config: dict[str, Any],
) -> None:
    rows: list[np.ndarray] = []
    for stem in stems:
        image, line = _read_pair(image_dir, label_dir, stem)
        raw = segment_grayworld_hsv(image)
        morphed = (
            raw.copy()
            if frozen_config.get("order") is None
            else apply_morphology(
                raw,
                open_kernel=frozen_config["open_kernel"],
                close_kernel=frozen_config["close_kernel"],
                order=str(frozen_config["order"]),
            )
        )
        cleaned = clean_candidate_mask(raw, frozen_config)
        overlay = cv2.addWeighted(
            image, 0.75, cv2.applyColorMap(line, cv2.COLORMAP_JET), 0.25, 0
        )
        rows.append(
            np.hstack(
                [
                    _color_tile(image, f"{stem} input"),
                    _gray_tile(raw, "Day61 raw"),
                    _gray_tile(morphed, "morphology"),
                    _gray_tile(cleaned, "region filtered"),
                    _color_tile(_region_overlay(image, cleaned), "contours + boxes"),
                    _color_tile(overlay, "row-line proxy"),
                ]
            )
        )
    if not rows or not cv2.imwrite(str(path), np.vstack(rows)):
        raise ValueError(f"cannot write contact sheet: {path}")


def _write_v2_contact_sheet(
    path: Path,
    image_dir: Path,
    label_dir: Path,
    stems: list[str],
    v2_config: dict[str, Any],
) -> None:
    rows: list[np.ndarray] = []
    for stem in stems:
        image, line = _read_pair(image_dir, label_dir, stem)
        raw = segment_grayworld_hsv(image)
        v1 = clean_candidate_mask(raw, V1_FROZEN_CONFIG)
        v2 = clean_candidate_mask(raw, v2_config)
        overlay = cv2.addWeighted(
            image, 0.75, cv2.applyColorMap(line, cv2.COLORMAP_JET), 0.25, 0
        )
        rows.append(
            np.hstack(
                [
                    _color_tile(image, f"{stem} input"),
                    _gray_tile(raw, "Day61 raw"),
                    _gray_tile(v1, "Day62 v1 fixed"),
                    _gray_tile(v2, "Day62 v2 perspective"),
                    _color_tile(_region_overlay(image, v2), "v2 contours + boxes"),
                    _color_tile(overlay, "row-line proxy"),
                ]
            )
        )
    if not rows or not cv2.imwrite(str(path), np.vstack(rows)):
        raise ValueError(f"cannot write v2 contact sheet: {path}")


def run_study(
    train_root: Path,
    train_manifest: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Run search/review/stress selection, then one frozen validation evaluation."""
    train_image_dir, train_label_dir = train_root / "image", train_root / "label"
    val_image_dir, val_label_dir = validation_root / "image", validation_root / "label"
    for path in (train_image_dir, train_label_dir, val_image_dir, val_label_dir):
        if not path.is_dir():
            raise ValueError(f"missing dataset directory: {path}")

    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(
        validation_manifest, "validation_development"
    )
    split = split_development_stems(train_stems)
    configs_by_name = {
        config["name"]: config for config in [RAW_CONFIG, *MORPHOLOGY_CONFIGS]
    }

    search_summaries, _ = evaluate_configurations(
        train_image_dir,
        train_label_dir,
        split["search"],
        [RAW_CONFIG, *MORPHOLOGY_CONFIGS],
    )
    search_baseline = search_summaries[RAW_CONFIG["name"]]
    search_ranked = _eligible_ranked(
        search_baseline,
        [search_summaries[config["name"]] for config in MORPHOLOGY_CONFIGS],
    )
    shortlist_names = [row["name"] for row in search_ranked[:4]]

    if shortlist_names:
        review_configs = [RAW_CONFIG] + [configs_by_name[name] for name in shortlist_names]
        review_summaries, _ = evaluate_configurations(
            train_image_dir, train_label_dir, split["review"], review_configs
        )
        review_baseline = review_summaries[RAW_CONFIG["name"]]
        selected_summary, review_checks = choose_candidate(
            review_baseline,
            [review_summaries[name] for name in shortlist_names],
        )
        selected_name = selected_summary["name"]
    else:
        review_summaries = {RAW_CONFIG["name"]: search_baseline}
        selected_name = RAW_CONFIG["name"]
        review_checks = {"no_search_candidate_passed": False}

    stress_configs = [RAW_CONFIG]
    if selected_name != RAW_CONFIG["name"]:
        stress_configs.append(configs_by_name[selected_name])
    stress_summaries, stress_rows = evaluate_configurations(
        train_image_dir, train_label_dir, split["stress"], stress_configs
    )
    stress_baseline = stress_summaries[RAW_CONFIG["name"]]
    if selected_name == RAW_CONFIG["name"]:
        frozen_name = RAW_CONFIG["name"]
        stress_checks = {"no_review_candidate_passed": False}
    else:
        stress_checks = acceptance_checks(
            stress_baseline, stress_summaries[selected_name]
        )
        frozen_name = (
            selected_name if all(stress_checks.values()) else RAW_CONFIG["name"]
        )
    frozen_config = configs_by_name[frozen_name]

    validation_configs = [RAW_CONFIG]
    if frozen_name != RAW_CONFIG["name"]:
        validation_configs.append(frozen_config)
    validation_summaries, validation_rows = evaluate_configurations(
        val_image_dir, val_label_dir, validation_stems, validation_configs
    )
    validation_baseline = validation_summaries[RAW_CONFIG["name"]]
    if frozen_name == RAW_CONFIG["name"]:
        validation_checks = {"morphology_not_accepted_before_validation": False}
        frozen_validation_summary = validation_baseline
    else:
        frozen_validation_summary = validation_summaries[frozen_name]
        validation_checks = acceptance_checks(
            validation_baseline, frozen_validation_summary
        )
    absolute_checks = {
        "mean_gap_at_least_0_20": frozen_validation_summary["mean_gap"] >= 0.20,
        "p10_gap_at_least_0_08": frozen_validation_summary["p10_gap"] >= 0.08,
        "max_off_line_at_most_0_55": (
            frozen_validation_summary["max_off_line_activation"] <= 0.55
        ),
    }
    morphology_accepted = (
        frozen_name != RAW_CONFIG["name"]
        and all(stress_checks.values())
        and all(validation_checks.values())
        and all(absolute_checks.values())
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = [
        {"partition": "stress", **row} for row in stress_rows
    ] + [{"partition": "validation", **row} for row in validation_rows]
    _write_csv(output_dir / "proxy_region_metrics.csv", output_rows)
    baseline_validation_rows = [
        row for row in validation_rows if row["configuration"] == RAW_CONFIG["name"]
    ]
    candidate_validation_rows = [
        row for row in validation_rows if row["configuration"] == frozen_name
    ]
    if frozen_name == RAW_CONFIG["name"]:
        candidate_validation_rows = baseline_validation_rows
    review_stems = _select_review_stems(
        baseline_validation_rows,
        candidate_validation_rows,
        min(comparison_count, len(validation_stems)),
    )
    _write_contact_sheet(
        output_dir / "comparison_contact_sheet.jpg",
        val_image_dir,
        val_label_dir,
        review_stems,
        frozen_config,
    )

    result = {
        "schema_version": 1,
        "marker": "DAY62_LESSON_COMPLETE",
        "scope": "CRDLD same-source positive development only; no external or reject-aware claim",
        "day61_input": {
            "method": "bounded_grayworld_plus_hsv",
            "gain_range": [0.5, 2.0],
            "hsv": {"h_min": 20, "h_max": 90, "s_min": 10, "v_min": 25},
            "retuned_on_day62": False,
        },
        "development_partitions": {name: len(ids) for name, ids in split.items()},
        "partition_rule": "salted SHA-256; search, review, and stress are disjoint",
        "candidate_configurations": [RAW_CONFIG, *MORPHOLOGY_CONFIGS],
        "search_summaries": search_summaries,
        "search_shortlist": shortlist_names,
        "review_summaries": review_summaries,
        "review_selected": selected_name,
        "review_checks": review_checks,
        "stress_summaries": stress_summaries,
        "stress_checks": stress_checks,
        "frozen_configuration_before_validation": frozen_config,
        "validation_count": len(validation_stems),
        "validation_summaries": validation_summaries,
        "validation_checks": validation_checks,
        "validation_absolute_checks": absolute_checks,
        "morphology_accepted_for_day63": morphology_accepted,
        "day63_input": frozen_name,
        "review_stems": review_stems,
        "metric_limitation": "Centerline proxy metrics are not vegetation IoU, Dice, precision, or false-positive rate.",
        "validation_accessed_only_after_freeze": True,
        "same_source_internal_benchmark_accessed": False,
        "frozen_external_accessed": False,
        "generalization_status": "BLOCKED_UNTIL_FROZEN_EXTERNAL_TEST",
    }
    (output_dir / "day62_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_v2_study(
    train_root: Path,
    train_manifest: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Select a perspective-aware area filter by train-only five-fold stability."""
    train_image_dir, train_label_dir = train_root / "image", train_root / "label"
    val_image_dir, val_label_dir = validation_root / "image", validation_root / "label"
    for path in (train_image_dir, train_label_dir, val_image_dir, val_label_dir):
        if not path.is_dir():
            raise ValueError(f"missing dataset directory: {path}")

    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(
        validation_manifest, "validation_development"
    )
    folds = split_cv_folds(train_stems, folds=5)
    configs = [V1_FROZEN_CONFIG, *V2_DIRECTIONAL_CONFIGS]
    fold_summaries: list[dict[str, dict[str, Any]]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, stems in enumerate(folds):
        summaries, rows = evaluate_configurations(
            train_image_dir, train_label_dir, stems, configs
        )
        fold_summaries.append(summaries)
        fold_rows.extend(
            {"fold": fold_index, "partition": "train_cv", **row} for row in rows
        )

    candidate_names = [config["name"] for config in V2_DIRECTIONAL_CONFIGS]
    selected_name, pass_counts = select_v2_from_fold_summaries(
        fold_summaries,
        baseline_name=V1_FROZEN_CONFIG["name"],
        candidate_names=candidate_names,
    )
    configs_by_name = {config["name"]: config for config in configs}
    selected_config = configs_by_name[selected_name]

    aggregate_train_summaries: dict[str, dict[str, Any]] = {}
    for name in (V1_FROZEN_CONFIG["name"], selected_name):
        rows = [row for row in fold_rows if row["configuration"] == name]
        aggregate_train_summaries[name] = {"name": name, **_summarize_rows(rows)}
    train_checks = (
        v2_acceptance_checks(
            aggregate_train_summaries[V1_FROZEN_CONFIG["name"]],
            aggregate_train_summaries[selected_name],
        )
        if selected_name != V1_FROZEN_CONFIG["name"]
        else {"no_v2_candidate_reached_four_folds": False}
    )

    validation_summaries, validation_rows = evaluate_configurations(
        val_image_dir,
        val_label_dir,
        validation_stems,
        [V1_FROZEN_CONFIG, selected_config]
        if selected_name != V1_FROZEN_CONFIG["name"]
        else [V1_FROZEN_CONFIG],
    )
    validation_checks = (
        v2_acceptance_checks(
            validation_summaries[V1_FROZEN_CONFIG["name"]],
            validation_summaries[selected_name],
        )
        if selected_name != V1_FROZEN_CONFIG["name"]
        else {"v1_retained_before_confirmation": False}
    )
    v2_accepted = (
        selected_name != V1_FROZEN_CONFIG["name"]
        and pass_counts[selected_name] >= 4
        and all(train_checks.values())
        and all(validation_checks.values())
    )
    day63_config = selected_config if v2_accepted else V1_FROZEN_CONFIG

    v1_validation_rows = [
        row
        for row in validation_rows
        if row["configuration"] == V1_FROZEN_CONFIG["name"]
    ]
    candidate_validation_rows = [
        row for row in validation_rows if row["configuration"] == selected_name
    ]
    if not candidate_validation_rows:
        candidate_validation_rows = v1_validation_rows
    v1_by_stem = {row["stem"]: row for row in v1_validation_rows}
    candidate_by_stem = {row["stem"]: row for row in candidate_validation_rows}
    ranked_preservation = sorted(
        validation_stems,
        key=lambda stem: (
            candidate_by_stem[stem]["mean_band_support"]
            - v1_by_stem[stem]["mean_band_support"]
        ),
        reverse=True,
    )
    ranked_risk = list(reversed(ranked_preservation))
    review_stems: list[str] = []
    for stem in ranked_preservation[: comparison_count // 2] + ranked_risk:
        if stem not in review_stems:
            review_stems.append(stem)
        if len(review_stems) == comparison_count:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = fold_rows + [
        {"fold": "", "partition": "reused_validation", **row}
        for row in validation_rows
    ]
    _write_csv(output_dir / "proxy_region_metrics_v2.csv", output_rows)
    _write_v2_contact_sheet(
        output_dir / "comparison_contact_sheet_v2.jpg",
        val_image_dir,
        val_label_dir,
        review_stems,
        day63_config,
    )

    result = {
        "schema_version": 2,
        "marker": "DAY62_V2_LESSON_COMPLETE",
        "reason_for_v2": "v1 used isotropic morphology/global area filtering and lacked vertical geometry-readiness metrics",
        "scope": "CRDLD same-source positive development only; no external or reject-aware claim",
        "day61_input_retuned": False,
        "v1_baseline_configuration": V1_FROZEN_CONFIG,
        "rejected_preliminary_round": "perspective-only area thresholds did not produce enough confirmatory geometry gain",
        "v2_candidate_configurations": V2_DIRECTIONAL_CONFIGS,
        "selection_data": "CRDLD train-development only",
        "selection_protocol": "five deterministic disjoint folds; candidate must pass at least four folds",
        "fold_sizes": [len(fold) for fold in folds],
        "fold_summaries": fold_summaries,
        "fold_pass_counts": pass_counts,
        "selected_before_confirmation": selected_name,
        "aggregate_train_summaries": aggregate_train_summaries,
        "aggregate_train_checks": train_checks,
        "confirmation_data": "248 CRDLD validation-development images already used by v1; confirmatory reuse, not a new untouched test",
        "validation_summaries": validation_summaries,
        "validation_checks": validation_checks,
        "v2_accepted_for_day63": v2_accepted,
        "day63_input_configuration": day63_config,
        "review_stems": review_stems,
        "metric_limitation": "Centerline and vertical-band proxy metrics are not vegetation IoU, physical geometry accuracy, or control performance.",
        "same_source_internal_benchmark_accessed": False,
        "frozen_external_accessed": False,
        "generalization_status": "BLOCKED_UNTIL_FROZEN_EXTERNAL_TEST",
    }
    (output_dir / "day62_results_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--study", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    runner = run_v2_study if args.study == "v2" else run_study
    runner(
        args.train_root,
        args.train_manifest,
        args.validation_root,
        args.validation_manifest,
        args.output_dir,
        comparison_count=args.count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
