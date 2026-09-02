"""Day61 color spaces, vegetation indices, and illumination comparison.

This lesson reads CRDLD development images only. The annotated masks are crop-row
centerlines, not vegetation-area masks, so reported numbers are explicitly proxy
metrics rather than segmentation IoU/Dice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


HSV_LOWER = np.array([20, 15, 25], dtype=np.uint8)
HSV_UPPER = np.array([95, 255, 255], dtype=np.uint8)
LAB_A_MAX = 126
LAB_L_MIN = 30
EXG_FIXED_THRESHOLD = 0.03
GRAYWORLD_HSV_LOWER = np.array([20, 10, 25], dtype=np.uint8)
GRAYWORLD_HSV_UPPER = np.array([90, 255, 255], dtype=np.uint8)
GRAYWORLD_GAIN_MIN = 0.5
GRAYWORLD_GAIN_MAX = 2.0
HARD_REVIEW_STEMS = ["258", "795", "52", "214", "417", "207", "39", "380"]


def _check_bgr(image_bgr: np.ndarray) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("expected uint8 BGR image with shape HxWx3")


def normalized_exg(image_bgr: np.ndarray) -> np.ndarray:
    """Return illumination-reduced ExG = 2g-r-b using normalized BGR channels."""
    _check_bgr(image_bgr)
    image = image_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(image)
    channel_sum = b + g + r + np.finfo(np.float32).eps
    return (2.0 * g / channel_sum - r / channel_sum - b / channel_sum).astype(
        np.float32
    )


def gray_world_balance(image_bgr: np.ndarray) -> np.ndarray:
    """Reduce global color cast by equalizing BGR means with bounded gains."""
    _check_bgr(image_bgr)
    image_float = image_bgr.astype(np.float32)
    channel_means = image_float.reshape(-1, 3).mean(axis=0)
    target_mean = float(channel_means.mean())
    gains = np.clip(
        target_mean / np.maximum(channel_means, 1.0),
        GRAYWORLD_GAIN_MIN,
        GRAYWORLD_GAIN_MAX,
    )
    return np.clip(image_float * gains, 0, 255).astype(np.uint8)


def segment_grayworld_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Primary v4 candidate: bounded Gray-World correction followed by fixed HSV."""
    corrected = gray_world_balance(image_bgr)
    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, GRAYWORLD_HSV_LOWER, GRAYWORLD_HSV_UPPER)


def read_manifest_stems(manifest_path: Path, expected_role: str) -> list[str]:
    """Read item IDs from an audited JSONL manifest and enforce its evidence role."""
    stems: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") != expected_role:
            raise ValueError(f"unexpected manifest role: {row.get('role')}")
        stems.append(str(row["item_id"]))
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("manifest item IDs must be non-empty and unique")
    return stems


def segment_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Fixed development-only green candidate using OpenCV HSV ranges."""
    _check_bgr(image_bgr)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)


def segment_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Green candidate from Lab a-channel with a minimal dark-pixel guard."""
    _check_bgr(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, _ = cv2.split(lab)
    return np.where((a_channel <= LAB_A_MAX) & (l_channel >= LAB_L_MIN), 255, 0).astype(
        np.uint8
    )


def segment_exg_otsu(image_bgr: np.ndarray) -> np.ndarray:
    """Rejected baseline: Otsu can force a foreground split on neutral soil."""
    exg = normalized_exg(image_bgr)
    exg_u8 = np.clip((exg + 1.0) / 3.0 * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_exg_fixed(
    image_bgr: np.ndarray, threshold: float = EXG_FIXED_THRESHOLD
) -> np.ndarray:
    """Fixed normalized-ExG threshold; unlike Otsu it may return no foreground."""
    return np.where(normalized_exg(image_bgr) >= threshold, 255, 0).astype(np.uint8)


def deterministic_experiment_split(
    stems: list[str],
    hard_stems: list[str],
    calibration_count: int = 32,
    review_count: int = 40,
    diagnostic_count: int = 64,
    verification_count: int = 64,
) -> dict[str, list[str]]:
    """Create deterministic, disjoint development partitions by salted hashes."""
    unique = set(stems)
    hard = [stem for stem in hard_stems if stem in unique]
    if len(unique) < calibration_count + review_count + diagnostic_count + verification_count:
        raise ValueError("not enough unique images for requested experiment split")

    def ranked(salt: str, excluded: set[str]) -> list[str]:
        candidates = unique - excluded
        return sorted(
            candidates,
            key=lambda stem: (
                hashlib.sha256(f"{salt}:{stem}".encode()).hexdigest(),
                int(stem),
            ),
        )

    calibration = ranked("calibration", set(hard))[:calibration_count]
    used = set(calibration)
    review_extra = ranked("review", used | set(hard))[: max(0, review_count - len(hard))]
    review = hard + review_extra
    used.update(review)
    diagnostic = ranked("verification", used)[:diagnostic_count]
    used.update(diagnostic)
    verification = ranked("final-verification", used)[:verification_count]
    return {
        "calibration": calibration,
        "review": review,
        "diagnostic": diagnostic,
        "verification": verification,
    }


def proxy_robustness_score(
    supports: list[float], off_line_activations: list[float]
) -> dict[str, float]:
    """Summarize row support, background activation, and lower-tail robustness."""
    support = np.asarray(supports, dtype=np.float64)
    off = np.asarray(off_line_activations, dtype=np.float64)
    if support.size == 0 or support.shape != off.shape:
        raise ValueError("support and off-line arrays must be non-empty and aligned")
    gaps = support - off
    mean_gap = float(gaps.mean())
    p10_gap = float(np.percentile(gaps, 10))
    mean_off = float(off.mean())
    max_off = float(off.max())
    score = (
        mean_gap
        + 0.2 * p10_gap
        - 1.5 * max(0.0, mean_off - 0.25)
        - 0.5 * max(0.0, max_off - 0.50)
    )
    return {
        "mean_support": float(support.mean()),
        "mean_off_line_activation": mean_off,
        "mean_gap": mean_gap,
        "p10_gap": p10_gap,
        "max_off_line_activation": max_off,
        "score": float(score),
    }


def compute_line_proxy_metrics(
    candidate_mask: np.ndarray, line_mask: np.ndarray, dilation_radius: int = 7
) -> dict[str, float]:
    """Measure vegetation activation around annotated row lines without calling it IoU."""
    if candidate_mask.shape != line_mask.shape:
        raise ValueError("candidate and line masks must have the same shape")
    candidate = candidate_mask > 0
    line = line_mask >= 128
    kernel_size = 2 * dilation_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    neighborhood = cv2.dilate(line.astype(np.uint8), kernel) > 0
    support = float(candidate[neighborhood].mean()) if neighborhood.any() else 0.0
    off_line = float(candidate[~neighborhood].mean()) if (~neighborhood).any() else 0.0
    return {
        "line_neighborhood_support": support,
        "off_line_activation": off_line,
        "candidate_fraction": float(candidate.mean()),
    }


def select_representative_samples(
    rows: list[dict[str, Any]], count: int = 8
) -> list[dict[str, Any]]:
    """Select deterministic lighting extremes while keeping sample IDs unique."""
    if len(rows) < count:
        raise ValueError(f"need at least {count} candidate images")
    mean_v_median = float(np.median([row["mean_v"] for row in rows]))
    shadow_median = float(np.median([row["shadow_fraction"] for row in rows]))
    selectors: list[tuple[str, Callable[[dict[str, Any]], float]]] = [
        ("darkest", lambda row: row["mean_v"]),
        ("brightest", lambda row: -row["mean_v"]),
        ("most_shadow", lambda row: -row["shadow_fraction"]),
        ("highest_contrast", lambda row: -row["contrast_v"]),
        ("lowest_saturation", lambda row: row["mean_s"]),
        ("highest_saturation", lambda row: -row["mean_s"]),
        ("median_brightness", lambda row: abs(row["mean_v"] - mean_v_median)),
        ("median_shadow", lambda row: abs(row["shadow_fraction"] - shadow_median)),
    ]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for reason, key in selectors:
        available = [row for row in rows if row["stem"] not in used]
        if not available or len(selected) >= count:
            break
        chosen = min(available, key=lambda row: (key(row), int(row["stem"])))
        chosen_with_reason = dict(chosen)
        chosen_with_reason["selection_reason"] = reason
        selected.append(chosen_with_reason)
        used.add(chosen["stem"])
    return selected


def _scan_images(image_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {path}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        rows.append(
            {
                "stem": path.stem,
                "mean_v": float(value.mean()),
                "shadow_fraction": float((value < 64).mean()),
                "contrast_v": float(value.std()),
                "mean_s": float(saturation.mean()),
            }
        )
    return rows


def _caption(tile: np.ndarray, text: str) -> np.ndarray:
    output = tile.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(output, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return output


def _gray_tile(channel: np.ndarray, title: str, size: int = 220) -> np.ndarray:
    tile = cv2.cvtColor(cv2.resize(channel, (size, size)), cv2.COLOR_GRAY2BGR)
    return _caption(tile, title)


def _color_tile(image: np.ndarray, title: str, size: int = 220) -> np.ndarray:
    return _caption(cv2.resize(image, (size, size)), title)


def _read_pair(image_dir: Path, label_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_dir / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    if image is None or label is None or image.shape[:2] != label.shape:
        raise ValueError(f"invalid pair {stem}")
    return image, np.where(label >= 128, 255, 0).astype(np.uint8)


def _mask_from_config(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if config["method"] == "grayworld_hsv":
        return segment_grayworld_hsv(image)
    if config["method"] == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(
            hsv,
            np.array([config["h_min"], config["s_min"], config["v_min"]], np.uint8),
            np.array([config["h_max"], 255, 255], np.uint8),
        )
    if config["method"] == "lab":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, _ = cv2.split(lab)
        return np.where(
            (a_channel <= config["a_max"]) & (l_channel >= config["l_min"]), 255, 0
        ).astype(np.uint8)
    if config["method"] == "exg_fixed":
        return segment_exg_fixed(image, config["threshold"])
    if config["method"] == "exg_otsu":
        return segment_exg_otsu(image)
    raise ValueError(f"unknown method {config['method']}")


def _evaluate_config(
    image_dir: Path, label_dir: Path, stems: list[str], config: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for stem in stems:
        image, line = _read_pair(image_dir, label_dir, stem)
        metrics = compute_line_proxy_metrics(_mask_from_config(image, config), line)
        rows.append({"stem": stem, "method": config["method"], **metrics})
    summary = proxy_robustness_score(
        [row["line_neighborhood_support"] for row in rows],
        [row["off_line_activation"] for row in rows],
    )
    summary["mean_candidate_fraction"] = float(
        np.mean([row["candidate_fraction"] for row in rows])
    )
    return summary, rows


def _summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary = proxy_robustness_score(
        [row["line_neighborhood_support"] for row in rows],
        [row["off_line_activation"] for row in rows],
    )
    summary["mean_candidate_fraction"] = float(
        np.mean([row["candidate_fraction"] for row in rows])
    )
    return summary


def _calibrate_methods(
    image_dir: Path, label_dir: Path, stems: list[str]
) -> dict[str, dict[str, Any]]:
    grids = {
        "hsv": [
            {"method": "hsv", "h_min": h, "h_max": 95, "s_min": s, "v_min": v}
            for h in (20, 25, 30)
            for s in (15, 25, 35, 45)
            for v in (15, 25, 35)
        ],
        "lab": [
            {"method": "lab", "a_max": a, "l_min": l}
            for a in (122, 124, 126, 128)
            for l in (20, 30, 40)
        ],
        "exg_fixed": [
            {"method": "exg_fixed", "threshold": threshold}
            for threshold in (-0.02, 0.0, 0.03, 0.05, 0.08, 0.10)
        ],
    }
    selected: dict[str, dict[str, Any]] = {}
    for method, configs in grids.items():
        candidates = []
        for config in configs:
            summary, _ = _evaluate_config(image_dir, label_dir, stems, config)
            candidates.append({"configuration": config, "calibration": summary})
        selected[method] = max(candidates, key=lambda row: row["calibration"]["score"])
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_v4_study(
    train_root: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Compare frozen v3 HSV with frozen Gray-World HSV on untouched validation data."""
    train_image_dir, train_label_dir = train_root / "image", train_root / "label"
    val_image_dir, val_label_dir = validation_root / "image", validation_root / "label"
    for path in (train_image_dir, train_label_dir, val_image_dir, val_label_dir):
        if not path.is_dir():
            raise ValueError(f"missing dataset directory: {path}")
    train_stems = [row["stem"] for row in _scan_images(train_image_dir)]
    validation_stems = read_manifest_stems(validation_manifest, "validation_development")
    baseline_config = {
        "method": "hsv", "h_min": 20, "h_max": 95, "s_min": 15, "v_min": 25
    }
    candidate_config = {
        "method": "grayworld_hsv",
        "gain_min": GRAYWORLD_GAIN_MIN,
        "gain_max": GRAYWORLD_GAIN_MAX,
        "h_min": int(GRAYWORLD_HSV_LOWER[0]),
        "h_max": int(GRAYWORLD_HSV_UPPER[0]),
        "s_min": int(GRAYWORLD_HSV_LOWER[1]),
        "v_min": int(GRAYWORLD_HSV_LOWER[2]),
    }
    train_baseline, _ = _evaluate_config(
        train_image_dir, train_label_dir, train_stems, baseline_config
    )
    train_candidate, _ = _evaluate_config(
        train_image_dir, train_label_dir, train_stems, candidate_config
    )

    validation_baseline, baseline_rows = _evaluate_config(
        val_image_dir, val_label_dir, validation_stems, baseline_config
    )
    validation_candidate, candidate_rows = _evaluate_config(
        val_image_dir, val_label_dir, validation_stems, candidate_config
    )
    absolute_checks = {
        "mean_gap_at_least_0_20": validation_candidate["mean_gap"] >= 0.20,
        "p10_gap_at_least_0_08": validation_candidate["p10_gap"] >= 0.08,
        "max_off_line_at_most_0_55": validation_candidate["max_off_line_activation"] <= 0.55,
    }
    relative_checks = {
        "score_gain_at_least_0_05": (
            validation_candidate["score"] - validation_baseline["score"] >= 0.05
        ),
        "max_off_reduction_at_least_0_10": (
            validation_baseline["max_off_line_activation"]
            - validation_candidate["max_off_line_activation"] >= 0.10
        ),
        "mean_gap_drop_at_most_0_01": (
            validation_candidate["mean_gap"] >= validation_baseline["mean_gap"] - 0.01
        ),
        "p10_not_worse": validation_candidate["p10_gap"] >= validation_baseline["p10_gap"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = (
        [{"partition": "validation", **row} for row in baseline_rows]
        + [{"partition": "validation", **row} for row in candidate_rows]
    )
    _write_csv(output_dir / "proxy_metrics_v4.csv", metric_rows)

    baseline_by_stem = {row["stem"]: row for row in baseline_rows}
    candidate_by_stem = {row["stem"]: row for row in candidate_rows}
    ranked_improvements = sorted(
        validation_stems,
        key=lambda stem: (
            baseline_by_stem[stem]["off_line_activation"]
            - candidate_by_stem[stem]["off_line_activation"]
        ),
        reverse=True,
    )
    worst_candidate = sorted(
        validation_stems,
        key=lambda stem: candidate_by_stem[stem]["off_line_activation"],
        reverse=True,
    )
    review_stems: list[str] = []
    for stem in ranked_improvements[: comparison_count // 2] + worst_candidate:
        if stem not in review_stems:
            review_stems.append(stem)
        if len(review_stems) == comparison_count:
            break
    contact_rows: list[np.ndarray] = []
    for stem in review_stems:
        image, line = _read_pair(val_image_dir, val_label_dir, stem)
        corrected = gray_world_balance(image)
        old_mask = _mask_from_config(image, baseline_config)
        new_mask = _mask_from_config(image, candidate_config)
        exg_mask = segment_exg_fixed(image)
        overlay = cv2.addWeighted(
            image, 0.75, cv2.applyColorMap(line, cv2.COLORMAP_JET), 0.25, 0
        )
        contact_rows.append(np.hstack([
            _color_tile(image, f"{stem} input"),
            _color_tile(corrected, "Gray-World"),
            _gray_tile(old_mask, "v3 HSV"),
            _gray_tile(new_mask, "v4 GW+HSV"),
            _gray_tile(exg_mask, "fixed ExG"),
            _color_tile(overlay, "row-line annotation"),
        ]))
    cv2.imwrite(str(output_dir / "comparison_contact_sheet_v4.jpg"), np.vstack(contact_rows))

    result = {
        "schema_version": 4,
        "marker": "DAY61_LESSON_COMPLETE",
        "scope": "CRDLD same-source positive development only; no external claim",
        "development_count": len(train_stems),
        "untouched_validation_count": len(validation_stems),
        "baseline_configuration": baseline_config,
        "candidate_configuration": candidate_config,
        "development_summary": {"v3_hsv": train_baseline, "v4_grayworld_hsv": train_candidate},
        "validation_summary": {
            "v3_hsv": validation_baseline,
            "v4_grayworld_hsv": validation_candidate,
        },
        "absolute_checks": absolute_checks,
        "relative_improvement_checks": relative_checks,
        "v4_replaces_v3_for_day62": all(absolute_checks.values()) and all(relative_checks.values()),
        "review_stems": review_stems,
        "metric_limitation": "Centerline proxy metrics are not vegetation IoU, Dice, precision, or false-positive rate.",
        "frozen_external_accessed": False,
        "generalization_status": "BLOCKED_UNTIL_FROZEN_EXTERNAL_TEST",
    }
    (output_dir / "day61_results_v4.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_lesson(train_root: Path, output_dir: Path, count: int = 8) -> dict[str, Any]:
    image_dir, label_dir = train_root / "image", train_root / "label"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError("train root must contain image/ and label/ directories")
    scan_rows = _scan_images(image_dir)
    split = deterministic_experiment_split(
        [row["stem"] for row in scan_rows], HARD_REVIEW_STEMS
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrated = _calibrate_methods(image_dir, label_dir, split["calibration"])
    review_rows: list[dict[str, Any]] = []
    review_summaries: dict[str, dict[str, float]] = {}
    for method, record in calibrated.items():
        summary, rows = _evaluate_config(
            image_dir, label_dir, split["review"], record["configuration"]
        )
        review_summaries[method] = summary
        review_rows.extend({"partition": "review", **row} for row in rows)

    # Method choice is frozen before the verification partition is evaluated.
    primary_method = max(review_summaries, key=lambda name: review_summaries[name]["score"])
    primary_config = calibrated[primary_method]["configuration"]
    diagnostic_summary, diagnostic_rows = _evaluate_config(
        image_dir, label_dir, split["diagnostic"], primary_config
    )
    primary_review_rows = [
        row for row in review_rows if row["method"] == primary_method
    ]
    candidate_fraction_upper = min(
        0.55, max(row["candidate_fraction"] for row in primary_review_rows) + 0.02
    )
    raw_verification_summary, verification_rows = _evaluate_config(
        image_dir, label_dir, split["verification"], primary_config
    )
    accepted_verification_rows = [
        row
        for row in verification_rows
        if row["candidate_fraction"] <= candidate_fraction_upper
    ]
    rejected_verification_stems = [
        row["stem"]
        for row in verification_rows
        if row["candidate_fraction"] > candidate_fraction_upper
    ]
    verification_summary = _summarize_metric_rows(accepted_verification_rows)
    rejection_rate = len(rejected_verification_stems) / len(verification_rows)
    acceptance = {
        "mean_gap_at_least_0_20": verification_summary["mean_gap"] >= 0.20,
        "p10_gap_at_least_0_08": verification_summary["p10_gap"] >= 0.08,
        "max_off_line_at_most_0_55": verification_summary["max_off_line_activation"] <= 0.55,
        "rejection_rate_at_most_0_10": rejection_rate <= 0.10,
    }

    otsu_summary, otsu_rows = _evaluate_config(
        image_dir,
        label_dir,
        split["verification"],
        {"method": "exg_otsu"},
    )
    all_rows = (
        review_rows
        + [{"partition": "diagnostic", **row} for row in diagnostic_rows]
        + [{"partition": "verification", **row} for row in verification_rows + otsu_rows]
    )
    _write_csv(output_dir / "proxy_metrics_v3.csv", all_rows)

    comparison_rows: list[np.ndarray] = []
    for stem in HARD_REVIEW_STEMS[:count]:
        image, line = _read_pair(image_dir, label_dir, stem)
        masks = {
            method: _mask_from_config(image, record["configuration"])
            for method, record in calibrated.items()
        }
        masks["exg_otsu"] = segment_exg_otsu(image)
        line_color = cv2.applyColorMap(line, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.75, line_color, 0.25, 0)
        comparison_rows.append(
            np.hstack(
                [
                    _color_tile(image, f"{stem} input"),
                    _gray_tile(masks["hsv"], "tuned HSV"),
                    _gray_tile(masks["lab"], "tuned Lab"),
                    _gray_tile(masks["exg_fixed"], "fixed ExG"),
                    _gray_tile(masks["exg_otsu"], "rejected Otsu"),
                    _color_tile(overlay, "row-line annotation"),
                ]
            )
        )
    cv2.imwrite(str(output_dir / "comparison_contact_sheet_v3.jpg"), np.vstack(comparison_rows))
    (output_dir / "experiment_split_v3.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = {
        "schema_version": 3,
        "marker": "DAY61_LESSON_COMPLETE",
        "scope": "CRDLD train development source only; no external-generalization claim",
        "partitions": {name: len(stems) for name, stems in split.items()},
        "partition_rule": "salted SHA-256, disjoint; final verification evaluated once after v3 method and guard freeze",
        "calibrated_candidates": calibrated,
        "review_summaries": review_summaries,
        "frozen_primary_method": primary_method,
        "frozen_primary_configuration": primary_config,
        "v2_diagnostic_failure_summary": diagnostic_summary,
        "failure_guard": {
            "rule": "reject if candidate_fraction exceeds review maximum plus 0.02, capped at 0.55",
            "candidate_fraction_upper": candidate_fraction_upper,
        },
        "raw_verification_summary": raw_verification_summary,
        "verification_summary": verification_summary,
        "verification_rejected_stems": rejected_verification_stems,
        "verification_rejection_rate": rejection_rate,
        "verification_acceptance": acceptance,
        "development_baseline_accepted_for_day62": all(acceptance.values()),
        "rejected_baseline": {
            "method": "exg_otsu",
            "reason": "per-image Otsu can force a foreground class on neutral soil",
            "verification_summary": otsu_summary,
        },
        "metric_limitation": "CRDLD labels are row centerlines, not vegetation-area masks; proxy values are not IoU, Dice, precision, or false-positive rate.",
        "frozen_external_accessed": False,
        "generalization_status": "BLOCKED_UNTIL_FROZEN_EXTERNAL_TEST",
    }
    (output_dir / "day61_results_v3.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    args = parser.parse_args()
    if bool(args.validation_root) != bool(args.validation_manifest):
        parser.error("--validation-root and --validation-manifest must be provided together")
    if args.validation_root:
        run_v4_study(
            args.train_root,
            args.validation_root,
            args.validation_manifest,
            args.output_dir,
            comparison_count=args.count,
        )
    else:
        run_lesson(args.train_root, args.output_dir, count=args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
