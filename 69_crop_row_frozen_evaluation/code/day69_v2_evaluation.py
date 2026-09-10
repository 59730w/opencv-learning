"""Day69 v2 partial-polyline evaluation with fail-closed summaries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any, Sequence

import numpy as np
import cv2

from day63_crop_row_geometry import CropRowLine


MODEL_Y_MIN = 0.40
MODEL_Y_MAX = 0.90
V2_ABSOLUTE_GATES = {
    "row_detection_precision": (">=", 0.80),
    "row_detection_recall": (">=", 0.80),
    "matched_visible_position_mae_norm": ("<=", 0.05),
    "runtime_median_ms_per_frame": ("<=", 50.0),
}


@dataclass(frozen=True)
class VisiblePolyline:
    x_norm: tuple[float, ...]
    y_norm: tuple[float, ...]

    @property
    def y_min(self) -> float:
        return self.y_norm[0]

    @property
    def y_max(self) -> float:
        return self.y_norm[-1]

    def x_at(self, y_norm: np.ndarray) -> np.ndarray:
        return np.interp(y_norm, self.y_norm, self.x_norm)

    @property
    def heading_deg(self) -> float:
        low = max(MODEL_Y_MIN, self.y_min)
        high = min(MODEL_Y_MAX, self.y_max)
        endpoints = self.x_at(np.asarray([low, high]))
        return float(np.degrees(np.arctan2(endpoints[0] - endpoints[1], high - low)))


def visible_reference_polylines(
    payload: dict[str, Any],
    *,
    width: int,
    height: int,
    min_overlap_y_norm: float,
) -> tuple[tuple[VisiblePolyline, ...], int]:
    """Keep labels with enough visible support inside the model's y domain."""
    if width < 2 or height < 2:
        raise ValueError("reference dimensions must be at least 2x2")
    if not 0.0 < min_overlap_y_norm <= MODEL_Y_MAX - MODEL_Y_MIN:
        raise ValueError("min_overlap_y_norm is outside the model evaluation domain")
    references: list[VisiblePolyline] = []
    rejected = 0
    for label in payload.get("labels", []):
        xs = np.asarray(label.get("x", []), dtype=np.float64)
        ys = np.asarray(label.get("y", []), dtype=np.float64)
        valid = (
            xs.size >= 2
            and xs.shape == ys.shape
            and np.all(np.isfinite(xs))
            and np.all(np.isfinite(ys))
        )
        if not valid:
            rejected += 1
            continue
        order = np.argsort(ys)
        y_sorted = ys[order] / (height - 1)
        x_sorted = xs[order] / (width - 1)
        unique_y, unique_indices = np.unique(y_sorted, return_index=True)
        unique_x = x_sorted[unique_indices]
        overlap = min(MODEL_Y_MAX, float(unique_y[-1])) - max(
            MODEL_Y_MIN, float(unique_y[0])
        )
        if unique_y.size < 2 or overlap + 1e-12 < min_overlap_y_norm:
            rejected += 1
            continue
        references.append(
            VisiblePolyline(
                x_norm=tuple(float(value) for value in unique_x),
                y_norm=tuple(float(value) for value in unique_y),
            )
        )
    references.sort(
        key=lambda row: float(row.x_at(np.asarray([max(MODEL_Y_MIN, row.y_min)]))[0])
    )
    return tuple(references), rejected


def _polygon_centerline(points: np.ndarray, *, sample_count: int = 33) -> VisiblePolyline | None:
    low = max(MODEL_Y_MIN, float(points[:, 1].min()))
    high = min(MODEL_Y_MAX, float(points[:, 1].max()))
    if high <= low:
        return None
    y_values: list[float] = []
    x_values: list[float] = []
    closed = np.vstack((points, points[0]))
    for y_norm in np.linspace(low, high, sample_count):
        intersections: list[float] = []
        for start, end in zip(closed[:-1], closed[1:]):
            y0, y1 = float(start[1]), float(end[1])
            if abs(y1 - y0) < 1e-12 or y_norm < min(y0, y1) or y_norm > max(y0, y1):
                continue
            fraction = (y_norm - y0) / (y1 - y0)
            if -1e-12 <= fraction <= 1.0 + 1e-12:
                intersections.append(float(start[0] + fraction * (end[0] - start[0])))
        unique = np.unique(np.round(intersections, 12))
        if unique.size >= 2:
            y_values.append(float(y_norm))
            x_values.append(float((unique[0] + unique[-1]) / 2.0))
    if len(y_values) < 2:
        return None
    return VisiblePolyline(tuple(x_values), tuple(y_values))


def yolo_segmentation_references(
    label_text: str, *, min_overlap_y_norm: float
) -> tuple[tuple[VisiblePolyline, ...], int]:
    """Convert normalized YOLO polygons to analytic scanline centerlines."""
    references: list[VisiblePolyline] = []
    rejected = 0
    for raw_line in label_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            values = np.asarray([float(value) for value in raw_line.split()], dtype=np.float64)
        except ValueError:
            rejected += 1
            continue
        coordinates = values[1:]
        if (
            values.size < 7
            or coordinates.size % 2
            or not np.all(np.isfinite(coordinates))
            or np.any(coordinates < 0.0)
            or np.any(coordinates > 1.0)
        ):
            rejected += 1
            continue
        reference = _polygon_centerline(coordinates.reshape(-1, 2))
        if reference is None or reference.y_max - reference.y_min + 1e-12 < min_overlap_y_norm:
            rejected += 1
            continue
        references.append(reference)
    references.sort(
        key=lambda row: float(row.x_at(np.asarray([max(MODEL_Y_MIN, row.y_min)]))[0])
    )
    return tuple(references), rejected


def _prediction_x_at(row: CropRowLine, y_norm: np.ndarray) -> np.ndarray:
    fraction = (y_norm - MODEL_Y_MIN) / (MODEL_Y_MAX - MODEL_Y_MIN)
    return row.far_x_norm + fraction * (row.near_x_norm - row.far_x_norm)


def match_visible_polylines(
    predictions: Sequence[CropRowLine],
    references: Sequence[VisiblePolyline],
    *,
    max_mean_x_error_norm: float,
    sample_count: int = 9,
) -> dict[str, Any]:
    """Monotonically match rows using mean x error on each label's visible support."""
    if max_mean_x_error_norm <= 0.0 or sample_count < 2:
        raise ValueError("invalid visible-overlap matching configuration")
    predictions = tuple(predictions)
    references = tuple(references)
    candidates: dict[tuple[int, int], tuple[float, float]] = {}
    for pred_index, prediction in enumerate(predictions):
        for ref_index, reference in enumerate(references):
            low = max(MODEL_Y_MIN, reference.y_min)
            high = min(MODEL_Y_MAX, reference.y_max)
            if high <= low:
                continue
            y_samples = np.linspace(low, high, sample_count)
            error = float(
                np.mean(np.abs(_prediction_x_at(prediction, y_samples) - reference.x_at(y_samples)))
            )
            if error <= max_mean_x_error_norm:
                candidates[(pred_index, ref_index)] = (error, high - low)

    @lru_cache(maxsize=None)
    def solve(pred_index: int, ref_index: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if pred_index >= len(predictions) or ref_index >= len(references):
            return 0, 0.0, ()
        options = [solve(pred_index + 1, ref_index), solve(pred_index, ref_index + 1)]
        if (pred_index, ref_index) in candidates:
            tail_count, tail_cost, tail_pairs = solve(pred_index + 1, ref_index + 1)
            error, _ = candidates[(pred_index, ref_index)]
            options.append(
                (tail_count + 1, tail_cost + error, ((pred_index, ref_index),) + tail_pairs)
            )
        return min(options, key=lambda item: (-item[0], item[1], item[2]))

    _, _, selected_pairs = solve(0, 0)
    matches = [
        (pred_index, ref_index, *candidates[(pred_index, ref_index)])
        for pred_index, ref_index in selected_pairs
    ]
    return {
        "predicted_count": len(predictions),
        "reference_count": len(references),
        "matched_count": len(matches),
        "pairs": [[pred_index, ref_index] for pred_index, ref_index, _, _ in matches],
        "mean_x_error_norm": statistics.fmean(item[2] for item in matches) if matches else None,
        "evaluated_overlap_y_norm": sum(item[3] for item in matches),
    }


_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)
_VIDEO_FRAME_SUFFIX = re.compile(r"(.+?)_mp4-\d+_jpg$", re.IGNORECASE)


def archive_video_group(entry_name: str) -> str:
    """Recover the source-video key preserved in a Roboflow archive filename."""
    leaf = str(entry_name).replace("\\", "/").rsplit("/", 1)[-1]
    base = _ROBOFLOW_SUFFIX.sub("", leaf)
    match = _VIDEO_FRAME_SUFFIX.fullmatch(base)
    return match.group(1) if match else base


def freeze_archive_groups(
    entry_names: Sequence[str],
    *,
    seed: str,
    minimum_group_images: int,
    reserved_group_fraction: float,
    maximum_images_per_group: int,
) -> dict[str, Any]:
    """Create a deterministic, group-disjoint holdout using filenames only."""
    if not seed:
        raise ValueError("seed must be non-empty")
    if minimum_group_images < 1 or maximum_images_per_group < 1:
        raise ValueError("group and sample sizes must be positive")
    if not 0.0 < reserved_group_fraction <= 1.0:
        raise ValueError("reserved_group_fraction must be in (0, 1]")
    by_group: dict[str, list[str]] = {}
    for name in sorted(set(str(value).replace("\\", "/") for value in entry_names)):
        if not name.lower().endswith(".jpg"):
            continue
        by_group.setdefault(archive_video_group(name), []).append(name)
    eligible = sorted(
        group for group, names in by_group.items() if len(names) >= minimum_group_images
    )
    if not eligible:
        raise ValueError("no source group meets minimum_group_images")
    ranked = sorted(
        eligible,
        key=lambda group: (hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), group),
    )
    reserved_count = max(1, round(len(ranked) * reserved_group_fraction))
    reserved_groups = ranked[:reserved_count]
    reserved_entries: list[str] = []
    for group in reserved_groups:
        ranked_entries = sorted(
            by_group[group],
            key=lambda name: (
                hashlib.sha256(f"{seed}:entry:{name}".encode()).hexdigest(),
                name,
            ),
        )
        reserved_entries.extend(ranked_entries[:maximum_images_per_group])
    return {
        "seed": seed,
        "minimum_group_images": minimum_group_images,
        "reserved_group_fraction": reserved_group_fraction,
        "maximum_images_per_group": maximum_images_per_group,
        "eligible_group_count": len(eligible),
        "reserved_groups": reserved_groups,
        "development_groups": sorted(set(eligible) - set(reserved_groups)),
        "reserved_entries": sorted(reserved_entries),
    }


def _gate_pass(value: float, operator: str, threshold: float) -> bool:
    return value >= threshold if operator == ">=" else value <= threshold


def geometry_summary_v2(
    records: Sequence[dict[str, Any]], *, evidence_role: str
) -> dict[str, Any]:
    """Aggregate geometry evidence; absent reference support invalidates model scores."""
    predicted = sum(int(row.get("predicted_count", 0)) for row in records)
    reference = sum(int(row.get("reference_count", 0)) for row in records)
    matched = sum(int(row.get("matched_count", 0)) for row in records)
    runtimes = [float(row["runtime_ms"]) for row in records if row.get("runtime_ms") is not None]
    evaluation_valid = reference > 0
    if evaluation_valid:
        precision = matched / predicted if predicted else 0.0
        recall = matched / reference
        position_error = (
            sum(float(row.get("visible_position_error_sum", 0.0)) for row in records) / matched
            if matched
            else None
        )
    else:
        precision = recall = position_error = None
    values = {
        "row_detection_precision": precision,
        "row_detection_recall": recall,
        "matched_visible_position_mae_norm": position_error,
        "runtime_median_ms_per_frame": statistics.median(runtimes) if runtimes else None,
    }
    gates = {
        name: {
            "value": value,
            "operator": V2_ABSOLUTE_GATES[name][0],
            "threshold": V2_ABSOLUTE_GATES[name][1],
            "passed": _gate_pass(value, *V2_ABSOLUTE_GATES[name]),
        }
        for name, value in values.items()
        if value is not None
    }
    return {
        "evidence_role": evidence_role,
        "evaluation_valid": evaluation_valid,
        "evaluation_status": "VALID" if evaluation_valid else "INVALID_REFERENCE_SUPPORT",
        "invalid_reason": None if evaluation_valid else "reference_row_count_is_zero",
        "record_count": len(records),
        "predicted_row_count": predicted,
        "reference_row_count": reference,
        "matched_row_count": matched,
        **values,
        "gates": gates,
        "all_supported_absolute_gates_passed": evaluation_valid
        and bool(gates)
        and all(item["passed"] is True for item in gates.values()),
    }


CENTRAL_ROW_GATES = {
    "central_row_detection_recall": (">=", 0.80),
    "matched_visible_position_mae_norm": ("<=", 0.05),
    "matched_heading_mae_deg": ("<=", 5.0),
}


def central_row_summary(
    records: Sequence[dict[str, Any]], *, evidence_role: str
) -> dict[str, Any]:
    """Aggregate single-central-row labels without treating other predicted rows as false positives."""
    references = sum(bool(row.get("reference_available")) for row in records)
    matched_records = [row for row in records if row.get("reference_available") and row.get("matched")]
    runtimes = [float(row["runtime_ms"]) for row in records if row.get("runtime_ms") is not None]
    valid = references > 0
    values: dict[str, float | None] = {
        "central_row_detection_recall": len(matched_records) / references if valid else None,
        "matched_visible_position_mae_norm": statistics.fmean(
            float(row["visible_position_error_norm"]) for row in matched_records
        ) if matched_records else None,
        "matched_heading_mae_deg": statistics.fmean(
            float(row["heading_error_deg"]) for row in matched_records
        ) if matched_records else None,
        "runtime_median_ms_per_frame": statistics.median(runtimes) if runtimes else None,
    }
    gates = {
        name: {
            "value": value,
            "operator": CENTRAL_ROW_GATES[name][0],
            "threshold": CENTRAL_ROW_GATES[name][1],
            "passed": _gate_pass(value, *CENTRAL_ROW_GATES[name]),
        }
        for name, value in values.items()
        if value is not None and name in CENTRAL_ROW_GATES
    }
    unavailable = [
        "all_row_detection_precision",
        "all_row_detection_recall",
        "corridor_boundary_pair_accuracy",
        "corridor_center_mae_norm",
        "unsafe_false_valid_rate",
        "reject_precision",
        "reject_recall",
        "comparable_640x360_runtime_gate",
    ]
    return {
        "evidence_role": evidence_role,
        "evaluation_valid": valid,
        "evaluation_status": "VALID" if valid else "INVALID_REFERENCE_SUPPORT",
        "invalid_reason": None if valid else "reference_image_count_is_zero",
        "record_count": len(records),
        "reference_image_count": references,
        "matched_reference_count": len(matched_records),
        "all_row_detection_precision": None,
        "all_row_detection_recall": None,
        **values,
        "runtime_gate_status": "UNAVAILABLE_MIXED_RESOLUTION_AND_WARMUP",
        "unavailable_metrics": unavailable,
        "gates": gates,
        "all_supported_absolute_gates_passed": valid
        and bool(gates)
        and all(item["passed"] is True for item in gates.values()),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_ssr_entries(
    entries: Sequence[dict[str, Any]],
    *,
    root: Path,
    predictor: Any,
    output_jsonl: Path,
    evidence_role: str,
    batch_size: int = 16,
    min_overlap_y_norm: float = 0.20,
    max_mean_x_error_norm: float = 0.06,
) -> dict[str, Any]:
    """Evaluate hash-frozen SSR image/polygon pairs as central-row-positive evidence."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root, output_jsonl = Path(root), Path(output_jsonl)
    records: list[dict[str, Any]] = []
    verified = 0
    for start in range(0, len(entries), batch_size):
        batch = list(entries[start : start + batch_size])
        frames: list[np.ndarray] = []
        references: list[tuple[tuple[VisiblePolyline, ...], int]] = []
        for entry in batch:
            image_path = root / str(entry["image_path"])
            label_path = root / str(entry["label_path"])
            for path, key in ((image_path, "image_sha256"), (label_path, "label_sha256")):
                actual = _file_sha256(path)
                expected = str(entry[key]).lower()
                if actual != expected:
                    raise ValueError(
                        f"{key} mismatch for {entry.get('item_id')}: expected {expected}, got {actual}"
                    )
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot decode image: {image_path}")
            reference = yolo_segmentation_references(
                label_path.read_text(encoding="utf-8"),
                min_overlap_y_norm=min_overlap_y_norm,
            )
            frames.append(frame)
            references.append(reference)
            verified += 1
        predictions = predictor.predict(frames)
        if len(predictions) != len(batch):
            raise ValueError("predictor must return one row sequence per image")
        runtime = float(getattr(predictor, "last_runtime_ms_per_frame", 0.0) or 0.0)
        for entry, predicted, (reference, rejected) in zip(batch, predictions, references):
            pred_rows = tuple(sorted(predicted, key=lambda row: row.far_x_norm))
            matching = match_visible_polylines(
                pred_rows,
                reference,
                max_mean_x_error_norm=max_mean_x_error_norm,
            )
            heading_errors = [
                abs(pred_rows[pred_index].heading_deg - reference[ref_index].heading_deg)
                for pred_index, ref_index in matching["pairs"]
            ]
            records.append(
                {
                    "item_id": str(entry.get("item_id", "")),
                    "evidence_role": evidence_role,
                    "predicted_row_count": len(pred_rows),
                    "reference_polygon_count": len(reference),
                    "reference_polygons_not_evaluable": rejected,
                    "reference_available": bool(reference),
                    "matched": matching["matched_count"] > 0,
                    "matched_count": matching["matched_count"],
                    "matched_pairs": matching["pairs"],
                    "visible_position_error_norm": matching["mean_x_error_norm"],
                    "heading_error_deg": statistics.fmean(heading_errors) if heading_errors else None,
                    "evaluated_overlap_y_norm": matching["evaluated_overlap_y_norm"],
                    "runtime_ms": runtime,
                }
            )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "evidence_role": evidence_role,
        "hash_verified_pair_count": verified,
        "output_jsonl": str(output_jsonl),
        "records_sha256": _file_sha256(output_jsonl),
        "summary": central_row_summary(records, evidence_role=evidence_role),
    }
