"""Day64 image coordinates for single-row history and current multi-row corridors.

The current revision converts ordered Day63 crop rows into adjacent corridor
boundaries, a corridor centerline, row spacing and a robust vanishing point.
Metric and robot-frame quantities remain blocked without calibration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import joblib
import numpy as np


DAY63_CODE_DIR = Path(__file__).resolve().parents[2] / "63_crop_row_geometry_extraction" / "code"
if str(DAY63_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DAY63_CODE_DIR))

from day63_crop_row_geometry import (  # noqa: E402
    CORRIDOR_AUDIT_Y_NORM,
    CropRowLine,
    ExtraTreesGeometryModel,
    FAR_Y_NORM,
    NEAR_Y_NORM,
    _frozen_day62_mask,
    _read_pair,
    central_label_geometry,
    decode_centerline_heatmap,
    derive_image_corridor,
    evaluate_prediction,
    extract_multirow_geometry,
    extract_multiscale_geometry_features,
    geometry_prediction_from_regression,
    match_ordered_crop_rows,
    read_manifest_stems,
)


@dataclass(frozen=True)
class ImageCoordinateMeasurement:
    status: str
    near_x_norm: float | None
    far_x_norm: float | None
    lateral_offset_norm: float | None
    lateral_offset_px: float | None
    heading_proxy_deg: float | None
    pixel_slope_deg: float | None
    horizon_y_norm: float
    horizon_intercept_norm: float | None
    confidence: float
    uncertainty: float | None
    vanishing_point_status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VanishingPointResult:
    status: str
    point_norm: tuple[float, float] | None
    residual_rms: float | None
    reason: str


@dataclass(frozen=True)
class CameraRayResult:
    status: str
    ray: tuple[float, float, float] | None
    reason: str


@dataclass(frozen=True)
class GroundMeasurementResult:
    status: str
    near_ground: tuple[float, float] | None
    far_ground: tuple[float, float] | None
    lateral_offset_m: float | None
    heading_deg: float | None
    reason: str


@dataclass(frozen=True)
class MultiRowCoordinateMeasurement:
    status: str
    row_count: int
    corridor_left_index: int | None
    corridor_right_index: int | None
    corridor_center_near_x_norm: float | None
    corridor_center_far_x_norm: float | None
    lateral_offset_norm: float | None
    lateral_offset_px: float | None
    heading_proxy_deg: float | None
    vanishing_point_status: str
    vanishing_point_norm: tuple[float, float] | None
    vanishing_point_residual_rms: float | None
    row_spacing_norm: tuple[float, ...]
    corridor_gap_ratio: float | None
    confidence: float
    camera_ray_status: str
    metric_measurement_status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_to_pixel(
    point_norm: tuple[float, float], *, width: int, height: int
) -> tuple[float, float]:
    """Convert inclusive [0, 1] image coordinates to pixel coordinates."""
    if width < 2 or height < 2:
        raise ValueError("width and height must be at least two pixels")
    x_norm, y_norm = map(float, point_norm)
    if not 0.0 <= x_norm <= 1.0 or not 0.0 <= y_norm <= 1.0:
        raise ValueError("normalized coordinates must lie in [0, 1]")
    return x_norm * (width - 1), y_norm * (height - 1)


def image_coordinate_measurement(
    *,
    near_x_norm: float | None,
    far_x_norm: float | None,
    width: int,
    height: int,
    input_status: str = "valid",
    confidence: float = 1.0,
    uncertainty: float | None = 0.0,
    horizon_y_norm: float = 0.25,
) -> ImageCoordinateMeasurement:
    """Convert one Day63 line to image-only offset and direction quantities."""
    if width < 2 or height < 2:
        raise ValueError("width and height must be at least two pixels")
    if input_status not in {"valid", "degraded", "reject"}:
        raise ValueError("input_status must be valid, degraded or reject")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1]")
    if uncertainty is not None and uncertainty < 0:
        raise ValueError("uncertainty must be non-negative")
    if not 0.0 <= horizon_y_norm <= 1.0:
        raise ValueError("horizon_y_norm must lie in [0, 1]")
    if input_status == "reject":
        return ImageCoordinateMeasurement(
            status="reject", near_x_norm=None, far_x_norm=None,
            lateral_offset_norm=None, lateral_offset_px=None,
            heading_proxy_deg=None, pixel_slope_deg=None,
            horizon_y_norm=horizon_y_norm, horizon_intercept_norm=None,
            confidence=confidence, uncertainty=uncertainty,
            vanishing_point_status="unavailable_single_line",
            reason="Day63 rejected geometry; no downstream measurement promoted")
    if near_x_norm is None or far_x_norm is None:
        raise ValueError("non-rejected geometry requires near and far endpoints")
    near = float(near_x_norm)
    far = float(far_x_norm)
    if not 0.0 <= near <= 1.0 or not 0.0 <= far <= 1.0:
        raise ValueError("line endpoints must lie in [0, 1]")
    forward_norm = NEAR_Y_NORM - FAR_Y_NORM
    heading_proxy = math.degrees(math.atan2(far - near, forward_norm))
    horizontal_px = (far - near) * (width - 1)
    forward_px = forward_norm * (height - 1)
    pixel_slope = math.degrees(math.atan2(horizontal_px, forward_px))
    horizon_x = near + (horizon_y_norm - NEAR_Y_NORM) * (
        far - near) / (FAR_Y_NORM - NEAR_Y_NORM)
    return ImageCoordinateMeasurement(
        status=input_status, near_x_norm=near, far_x_norm=far,
        lateral_offset_norm=near - 0.5,
        lateral_offset_px=(near - 0.5) * (width - 1),
        heading_proxy_deg=heading_proxy, pixel_slope_deg=pixel_slope,
        horizon_y_norm=horizon_y_norm, horizon_intercept_norm=float(horizon_x),
        confidence=confidence, uncertainty=uncertainty,
        vanishing_point_status="unavailable_single_line",
        reason="image-plane quantities only; one line cannot determine a true vanishing point")


def _normalized_line(
    first: tuple[float, float], second: tuple[float, float]
) -> np.ndarray:
    p1 = np.array([*first, 1.0], dtype=np.float64)
    p2 = np.array([*second, 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    scale = float(np.linalg.norm(line[:2]))
    if scale <= 1e-12:
        raise ValueError("line endpoints must be distinct")
    return line / scale


def estimate_vanishing_point(
    lines: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> VanishingPointResult:
    """Estimate a shared normalized intersection with robust IRLS line fitting."""
    if len(lines) < 2:
        return VanishingPointResult(
            "unavailable", None, None, "at_least_two_lines_required")
    coefficients = np.asarray(
        [_normalized_line(first, second) for first, second in lines],
        dtype=np.float64)
    design = coefficients[:, :2]
    target = -coefficients[:, 2]
    if np.linalg.matrix_rank(design, tol=1e-8) < 2:
        return VanishingPointResult(
            "unavailable", None, None, "lines_are_parallel_or_ill_conditioned")
    singular_values = np.linalg.svd(design, compute_uv=False)
    if singular_values[-1] <= 1e-8 or singular_values[0] / singular_values[-1] > 1e6:
        return VanishingPointResult(
            "unavailable", None, None, "lines_are_parallel_or_ill_conditioned")
    point = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(6):
        residuals = design @ point - target
        scale = max(float(np.median(np.abs(residuals))) * 1.4826, 1e-6)
        threshold = 1.5 * scale
        weights = np.minimum(1.0, threshold / np.maximum(np.abs(residuals), 1e-12))
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_target = target * np.sqrt(weights)
        point = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
    residuals = design @ point - target
    return VanishingPointResult(
        status="available", point_norm=(float(point[0]), float(point[1])),
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
        reason="robust intersection from two or more non-parallel image lines")


def multirow_coordinate_measurement(
    *,
    rows: Sequence[CropRowLine],
    width: int,
    height: int,
    input_status: str = "valid",
) -> MultiRowCoordinateMeasurement:
    """Convert ordered crop rows into an image corridor without metric claims."""
    if width < 2 or height < 2:
        raise ValueError("width and height must be at least two pixels")
    if input_status not in {"valid", "degraded", "reject"}:
        raise ValueError("input_status must be valid, degraded or reject")
    ordered = tuple(sorted(rows, key=lambda row: row.x_at(CORRIDOR_AUDIT_Y_NORM)))
    positions = np.asarray(
        [row.x_at(CORRIDOR_AUDIT_Y_NORM) for row in ordered], dtype=np.float64
    )
    spacings = tuple(map(float, np.diff(positions))) if len(positions) >= 2 else ()
    line_points = [
        ((row.near_x_norm, NEAR_Y_NORM), (row.far_x_norm, FAR_Y_NORM))
        for row in ordered
    ]
    vanishing = estimate_vanishing_point(line_points)
    corridor = derive_image_corridor(ordered)
    central_row_present = bool(
        len(positions) and np.any(np.abs(positions - 0.5) <= 0.04)
    )
    left_candidates = np.flatnonzero(positions < 0.5)
    right_candidates = np.flatnonzero(positions > 0.5)
    left_index = int(left_candidates[-1]) if len(left_candidates) else None
    right_index = int(right_candidates[0]) if len(right_candidates) else None
    corridor_gap_ratio = None
    if left_index is not None and right_index is not None and spacings:
        corridor_gap = positions[right_index] - positions[left_index]
        other_spacings = [
            spacing
            for index, spacing in enumerate(spacings)
            if index != left_index
        ]
        typical_spacing = float(
            np.median(other_spacings if other_spacings else spacings)
        )
        if typical_spacing > 1e-9:
            corridor_gap_ratio = float(corridor_gap / typical_spacing)

    if input_status == "reject" or not ordered:
        status = "reject"
        reason = "Day63 rejected rows; no downstream corridor measurement promoted"
    elif central_row_present:
        status = "degraded"
        reason = "central crop row occupies the camera corridor; no drivable center emitted"
    elif corridor is None or left_index is None or right_index is None:
        status = "degraded"
        reason = "both supported adjacent crop-row boundaries are required"
    elif vanishing.status != "available":
        status = "degraded"
        reason = "corridor exists but a stable multi-line vanishing point is unavailable"
    elif input_status == "degraded":
        status = "degraded"
        reason = "Day63 degraded input is preserved without promotion"
    else:
        status = "valid"
        reason = "adjacent crop-row boundaries define an image-space corridor"

    if status == "valid" and corridor is not None:
        near_x = corridor.near_x_norm
        far_x = corridor.far_x_norm
        heading = math.degrees(
            math.atan2(far_x - near_x, NEAR_Y_NORM - FAR_Y_NORM)
        )
        offset = near_x - 0.5
        confidence = min(
            ordered[left_index].confidence,
            ordered[right_index].confidence,
        )
    else:
        left_index = right_index = None
        near_x = far_x = heading = offset = None
        confidence = float(np.mean([row.confidence for row in ordered])) if ordered else 0.0
    return MultiRowCoordinateMeasurement(
        status=status,
        row_count=len(ordered),
        corridor_left_index=left_index,
        corridor_right_index=right_index,
        corridor_center_near_x_norm=near_x,
        corridor_center_far_x_norm=far_x,
        lateral_offset_norm=offset,
        lateral_offset_px=(offset * (width - 1)) if offset is not None else None,
        heading_proxy_deg=heading,
        vanishing_point_status=vanishing.status,
        vanishing_point_norm=vanishing.point_norm,
        vanishing_point_residual_rms=vanishing.residual_rms,
        row_spacing_norm=spacings,
        corridor_gap_ratio=corridor_gap_ratio,
        confidence=float(confidence),
        camera_ray_status="blocked_no_calibration",
        metric_measurement_status="blocked_no_ground_transform",
        reason=reason,
    )


def project_pixel_to_camera_ray(
    pixel: tuple[float, float],
    *,
    camera_matrix: np.ndarray | None,
) -> CameraRayResult:
    """Back-project a pixel with verified intrinsics; block when they are absent."""
    if camera_matrix is None:
        return CameraRayResult(
            "blocked_no_calibration", None, "camera intrinsic matrix is unavailable")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise ValueError("camera_matrix must be invertible")
    homogeneous = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    ray = np.linalg.solve(matrix, homogeneous)
    ray /= np.linalg.norm(ray)
    return CameraRayResult(
        "available_calibrated", tuple(map(float, ray)),
        "unit camera-frame ray from supplied intrinsic matrix")


def ground_plane_measurement(
    *,
    near_pixel: tuple[float, float],
    far_pixel: tuple[float, float],
    image_to_ground: np.ndarray | None,
    robot_center_x_m: float = 0.0,
) -> GroundMeasurementResult:
    """Map endpoints to declared X-right/Z-forward ground coordinates."""
    if image_to_ground is None:
        return GroundMeasurementResult(
            "blocked_no_ground_transform", None, None, None, None,
            "calibrated image-to-ground transform is unavailable")
    transform = np.asarray(image_to_ground, dtype=np.float64)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("image_to_ground must be a finite 3x3 matrix")
    points = np.asarray([[near_pixel, far_pixel]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(points, transform)[0]
    near_ground = tuple(map(float, mapped[0]))
    far_ground = tuple(map(float, mapped[1]))
    delta_x = far_ground[0] - near_ground[0]
    delta_forward = far_ground[1] - near_ground[1]
    if math.hypot(delta_x, delta_forward) <= 1e-12:
        raise ValueError("ground endpoints must be distinct")
    return GroundMeasurementResult(
        "available_synthetic_or_calibrated", near_ground, far_ground,
        near_ground[0] - float(robot_center_x_m),
        math.degrees(math.atan2(delta_x, delta_forward)),
        "requires supplied transform to define X-right/Z-forward metric coordinates")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    valid = [row for row in rows if row["status"] == "valid"]
    return {
        "count": float(len(rows)),
        "valid_fraction": float(np.mean([row["status"] == "valid" for row in rows])),
        "degraded_fraction": float(np.mean([row["status"] == "degraded" for row in rows])),
        "reject_fraction": float(np.mean([row["status"] == "reject" for row in rows])),
        "lateral_offset_abs_mean_norm": float(np.mean(
            [abs(row["lateral_offset_norm"]) for row in valid])) if valid else float("nan"),
        "heading_proxy_abs_mean_deg": float(np.mean(
            [abs(row["heading_proxy_deg"]) for row in valid])) if valid else float("nan"),
        "inherited_position_mae_norm": float(np.mean(
            [row["bottom_position_abs_error_norm"] for row in valid])) if valid else float("nan"),
        "inherited_heading_mae_deg": float(np.mean(
            [row["heading_abs_error_deg"] for row in valid])) if valid else float("nan"),
        "inherited_dual_threshold_fraction_all": float(np.mean(
            [row["within_both_absolute_thresholds"] for row in rows])),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _caption(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(output, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _measurement_overlay(
    image: np.ndarray, measurement: ImageCoordinateMeasurement, reference: Any,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    center_x = round(0.5 * (width - 1))
    cv2.line(output, (center_x, 0), (center_x, height - 1), (255, 255, 0), 1)

    def draw(near_x: float | None, far_x: float | None,
             color: tuple[int, int, int], thickness: int) -> None:
        if near_x is None or far_x is None:
            return
        near = normalized_to_pixel((near_x, NEAR_Y_NORM), width=width, height=height)
        far = normalized_to_pixel((far_x, FAR_Y_NORM), width=width, height=height)
        cv2.line(output, tuple(map(round, far)), tuple(map(round, near)),
                 color, thickness, cv2.LINE_AA)

    draw(reference.near_x_norm, reference.far_x_norm, (255, 0, 255), 2)
    draw(measurement.near_x_norm, measurement.far_x_norm, (0, 255, 255), 3)
    if measurement.lateral_offset_norm is not None:
        text = (f"offset={measurement.lateral_offset_norm:+.3f}  "
                f"heading={measurement.heading_proxy_deg:+.2f}deg")
        cv2.putText(output, text, (6, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _review_stems(rows: list[dict[str, Any]], count: int) -> list[str]:
    available = [row for row in rows if row["bottom_position_abs_error_norm"] is not None]
    if not available:
        return []
    by_position = sorted(available, key=lambda row: row["bottom_position_abs_error_norm"])
    by_heading = sorted(available, key=lambda row: row["heading_abs_error_deg"])
    by_uncertainty = sorted(available, key=lambda row: row["uncertainty"])
    candidates = [by_position[0], by_position[len(by_position) // 2], by_position[-1],
                  by_heading[-1], by_uncertainty[0], by_uncertainty[-1], *by_position]
    stems: list[str] = []
    for row in candidates:
        if row["stem"] not in stems:
            stems.append(row["stem"])
        if len(stems) == count:
            break
    return stems


def _write_contact_sheet(
    path: Path, samples: dict[str, dict[str, Any]], stems: list[str]
) -> None:
    rows: list[np.ndarray] = []
    for stem in stems:
        sample = samples[stem]
        mask_bgr = cv2.cvtColor(sample["mask"], cv2.COLOR_GRAY2BGR)
        overlay = _measurement_overlay(
            sample["image"], sample["measurement"], sample["reference"])
        status_panel = np.zeros_like(sample["image"])
        lines = [
            f"status: {sample['measurement'].status}",
            f"uncertainty: {sample['measurement'].uncertainty:.4f}",
            "VP: BLOCKED (single line)",
            "metric: BLOCKED (no calibration)",
        ]
        for index, text in enumerate(lines):
            cv2.putText(status_panel, text, (12, 45 + 32 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                        1, cv2.LINE_AA)
        tiles = [
            _caption(sample["image"], f"{stem} input"),
            _caption(mask_bgr, "frozen Day62 mask"),
            _caption(overlay, "cyan center / yellow pred / magenta GT"),
            _caption(status_panel, "measurement boundaries"),
        ]
        tiles = [cv2.resize(tile, (280, 220), interpolation=cv2.INTER_AREA)
                 for tile in tiles]
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


def run_day64_study(
    *,
    day63_model_path: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Apply the frozen Day63 model and audit Day64 coordinate definitions."""
    payload = joblib.load(day63_model_path)
    if payload.get("feature_length") != 320:
        raise ValueError("Day63 model feature contract is not the frozen 320-vector")
    model = ExtraTreesGeometryModel(estimator=payload["estimator"])
    image_dir = validation_root / "image"
    label_dir = validation_root / "label"
    stems = read_manifest_stems(validation_manifest, "validation_development")
    rows: list[dict[str, Any]] = []
    samples: dict[str, dict[str, Any]] = {}
    for stem in stems:
        image, label = _read_pair(image_dir, label_dir, stem)
        mask = _frozen_day62_mask(image)
        features = extract_multiscale_geometry_features(mask)
        endpoints, uncertainties = model.predict_with_uncertainty(features[None, :])
        geometry = geometry_prediction_from_regression(
            mask, endpoints=endpoints[0], uncertainty=float(uncertainties[0]),
            method=str(payload.get("selected_config", {}).get("name", "day63_frozen")))
        measurement = image_coordinate_measurement(
            near_x_norm=geometry.near_x_norm, far_x_norm=geometry.far_x_norm,
            width=image.shape[1], height=image.shape[0], input_status=geometry.status,
            confidence=geometry.confidence, uncertainty=float(uncertainties[0]))
        reference = central_label_geometry(label)
        metrics = evaluate_prediction(geometry, reference)
        heading_consistency = (
            abs(measurement.heading_proxy_deg - geometry.heading_deg)
            if measurement.heading_proxy_deg is not None and geometry.heading_deg is not None
            else None)
        row = {
            "stem": stem, "status": measurement.status,
            "confidence": measurement.confidence,
            "uncertainty": measurement.uncertainty,
            "near_x_norm": measurement.near_x_norm,
            "far_x_norm": measurement.far_x_norm,
            "lateral_offset_norm": measurement.lateral_offset_norm,
            "lateral_offset_px": measurement.lateral_offset_px,
            "heading_proxy_deg": measurement.heading_proxy_deg,
            "pixel_slope_deg": measurement.pixel_slope_deg,
            "horizon_intercept_norm": measurement.horizon_intercept_norm,
            "vanishing_point_status": measurement.vanishing_point_status,
            "camera_ray_status": "blocked_no_calibration",
            "metric_measurement_status": "blocked_no_ground_transform",
            "heading_consistency_abs_deg": heading_consistency,
            **metrics,
        }
        rows.append(row)
        samples[stem] = {"image": image, "mask": mask, "measurement": measurement,
                         "reference": reference}
    summary = _summarize(rows)
    consistency_values = [row["heading_consistency_abs_deg"] for row in rows
                          if row["heading_consistency_abs_deg"] is not None]
    consistency_max = float(max(consistency_values, default=0.0))
    gates = {
        "frozen_model_feature_contract_320": payload.get("feature_length") == 320,
        "day63_heading_reproduction_max_at_most_1e_12": consistency_max <= 1e-12,
        "all_real_outputs_mark_true_vanishing_point_unavailable": all(
            row["vanishing_point_status"] == "unavailable_single_line" for row in rows),
        "all_camera_rays_blocked_without_calibration": all(
            row["camera_ray_status"] == "blocked_no_calibration" for row in rows),
        "all_metric_measurements_blocked_without_ground_transform": all(
            row["metric_measurement_status"] == "blocked_no_ground_transform" for row in rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "coordinate_metrics.csv", rows)
    review_stems = _review_stems(rows, comparison_count)
    _write_contact_sheet(output_dir / "coordinate_contact_sheet.jpg", samples, review_stems)
    result = {
        "schema_version": 1,
        "marker": "DAY64_LESSON_COMPLETE",
        "scope": "image-coordinate definition and calibration-boundary audit on already-reused CRDLD positive development images",
        "day63_model_path": str(day63_model_path),
        "day63_selected_config": payload.get("selected_config"),
        "day63_model_retuned": False,
        "day61_color_retuned": False,
        "day62_morphology_retuned": False,
        "coordinate_contract": {
            "near_y_norm": NEAR_Y_NORM, "far_y_norm": FAR_Y_NORM,
            "lateral_offset_norm": "near_x_norm - 0.5; positive means row is right of image centre",
            "heading_proxy_deg": "atan2(far_x_norm-near_x_norm, near_y_norm-far_y_norm)",
            "horizon_intercept": "single-line extrapolation proxy, not a true vanishing point",
        },
        "summary": summary,
        "day63_heading_reproduction_max_abs_deg": consistency_max,
        "vanishing_point_on_real_outputs": "BLOCKED_SINGLE_LINE",
        "camera_calibration_available": False,
        "camera_ray_status": "BLOCKED_NO_CALIBRATION",
        "metric_measurement_available": False,
        "metric_measurement_status": "BLOCKED_NO_GROUND_TRANSFORM",
        "physical_measurement_blockers": [
            "camera intrinsics and distortion are unavailable",
            "camera-to-ground extrinsics or calibrated homography are unavailable",
            "one central line cannot establish a true vanishing point",
        ],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "review_stems": review_stems,
        "validation_is_untouched": False,
        "same_source_internal_benchmark_accessed": False,
        "frozen_external_accessed": False,
        "real_robot_control_established": False,
        "day65_handoff_status": "READY_FOR_TEMPORAL_STABILITY"
        if all(gates.values()) else "BLOCKED_BY_DAY64_GATE",
    }
    (output_dir / "day64_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _multirow_overlay(
    image: np.ndarray,
    prediction: MultiRowCoordinateMeasurement,
    reference: MultiRowCoordinateMeasurement,
    predicted_rows: Sequence[CropRowLine],
    reference_rows: Sequence[CropRowLine],
) -> np.ndarray:
    """Draw crop rows faintly and reserve thick lines for corridor boundaries."""
    output = image.copy()
    height, width = output.shape[:2]

    def point(row: CropRowLine, y_norm: float) -> tuple[int, int]:
        x_norm = float(np.clip(row.x_at(y_norm), 0.0, 1.0))
        return tuple(map(round, normalized_to_pixel(
            (x_norm, y_norm), width=width, height=height
        )))

    for row in reference_rows:
        cv2.line(output, point(row, FAR_Y_NORM), point(row, NEAR_Y_NORM),
                 (180, 0, 180), 1, cv2.LINE_AA)
    for row in predicted_rows:
        cv2.line(output, point(row, FAR_Y_NORM), point(row, NEAR_Y_NORM),
                 (0, 180, 180), 1, cv2.LINE_AA)
    for measurement, rows, color in (
        (reference, reference_rows, (255, 0, 255)),
        (prediction, predicted_rows, (0, 255, 255)),
    ):
        if measurement.corridor_left_index is not None:
            ordered = tuple(sorted(rows, key=lambda row: row.x_at(CORRIDOR_AUDIT_Y_NORM)))
            for index in (
                measurement.corridor_left_index,
                measurement.corridor_right_index,
            ):
                row = ordered[index]
                cv2.line(output, point(row, FAR_Y_NORM), point(row, NEAR_Y_NORM),
                         color, 3, cv2.LINE_AA)
        if measurement.corridor_center_near_x_norm is not None:
            center = CropRowLine(
                measurement.corridor_center_far_x_norm,
                measurement.corridor_center_near_x_norm,
                measurement.confidence,
                1,
            )
            cv2.line(output, point(center, FAR_Y_NORM), point(center, NEAR_Y_NORM),
                     (255, 255, 255) if color == (0, 255, 255) else color,
                     2, cv2.LINE_AA)
    if prediction.vanishing_point_norm is not None:
        vp_x, vp_y = prediction.vanishing_point_norm
        if -0.25 <= vp_x <= 1.25 and -0.25 <= vp_y <= 1.25:
            px = round(vp_x * (width - 1))
            py = round(vp_y * (height - 1))
            cv2.drawMarker(output, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    text = (
        f"pred={prediction.status} rows={prediction.row_count} "
        f"offset={prediction.lateral_offset_norm:+.3f}"
        if prediction.lateral_offset_norm is not None
        else f"pred={prediction.status} rows={prediction.row_count} no corridor"
    )
    cv2.putText(output, text, (6, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _finite_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _finite_median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _summarize_multirow_measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference_supported = [row for row in rows if row["reference_status"] == "valid"]
    unsafe_reference = [row for row in rows if row["reference_status"] != "valid"]
    paired = [row for row in rows if row["corridor_center_error_norm"] is not None]
    vp_paired = [row for row in rows if row["vanishing_point_error_norm"] is not None]
    boundary_rows = [row for row in rows if row["reference_status"] == "valid"]
    return {
        "frame_count": len(rows),
        "reference_supported_frame_count": len(reference_supported),
        "predicted_valid_fraction": float(np.mean(
            [row["prediction_status"] == "valid" for row in rows]
        )),
        "supported_valid_recall": float(np.mean(
            [row["prediction_status"] == "valid" for row in reference_supported]
        )) if reference_supported else None,
        "unsafe_false_valid_rate": float(np.mean(
            [row["prediction_status"] == "valid" for row in unsafe_reference]
        )) if unsafe_reference else None,
        "corridor_boundary_pair_accuracy": float(np.mean(
            [row["boundary_pair_correct"] for row in boundary_rows]
        )) if boundary_rows else None,
        "corridor_center_mae_norm": _finite_mean(
            [row["corridor_center_error_norm"] for row in paired]
        ),
        "corridor_heading_mae_deg": _finite_mean(
            [row["corridor_heading_error_deg"] for row in paired]
        ),
        "vanishing_point_available_fraction": float(np.mean(
            [row["prediction_vanishing_point_status"] == "available" for row in rows]
        )),
        "vanishing_point_median_error_norm": _finite_median(
            [row["vanishing_point_error_norm"] for row in vp_paired]
        ),
        "row_spacing_median_mae_norm": _finite_mean(
            [row["row_spacing_median_error_norm"] for row in rows
             if row["row_spacing_median_error_norm"] is not None]
        ),
    }


def _write_multirow_contact_sheet(
    path: Path,
    samples: list[dict[str, Any]],
    comparison_count: int,
) -> list[str]:
    ranked = sorted(
        samples,
        key=lambda sample: (
            sample["record"]["boundary_pair_correct"],
            -(sample["record"]["corridor_center_error_norm"] or 0.0),
        ),
    )
    if not ranked:
        return []
    indices = np.linspace(0, len(ranked) - 1, min(comparison_count, len(ranked)))
    selected = [ranked[round(index)] for index in indices]
    panels: list[np.ndarray] = []
    review_stems: list[str] = []
    for sample in selected:
        record = sample["record"]
        review_stems.append(record["partition"] + "/" + record["stem"])
        overlay = _multirow_overlay(
            sample["image"], sample["prediction_measurement"],
            sample["reference_measurement"], sample["prediction"].rows,
            sample["reference"].rows,
        )
        heatmap = cv2.applyColorMap(
            np.clip(sample["probability"] * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        heatmap = cv2.resize(
            heatmap, (sample["image"].shape[1], sample["image"].shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        status = np.zeros_like(sample["image"])
        lines = [
            f"rows pred/ref: {record['predicted_row_count']}/{record['reference_row_count']}",
            f"boundary pair: {record['boundary_pair_correct']}",
            f"VP: {record['prediction_vanishing_point_status']}",
            "metric: BLOCKED (no calibration)",
        ]
        for index, line in enumerate(lines):
            cv2.putText(status, line, (10, 38 + index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255),
                        1, cv2.LINE_AA)
        tiles = [
            _caption(sample["image"], review_stems[-1]),
            _caption(heatmap, "Day63 OOF centerline probability"),
            _caption(overlay, "yellow pred / magenta GT / white corridor"),
            _caption(status, "Day64 measurement audit"),
        ]
        tiles = [cv2.resize(tile, (300, 220), interpolation=cv2.INTER_AREA)
                 for tile in tiles]
        panels.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(panels))
    return review_stems


def run_day64_multirow_study(
    *,
    partitions: Sequence[dict[str, Any]],
    output_dir: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Audit Day63 OOF rows as Day64 image-space corridor measurements."""
    if not partitions:
        raise ValueError("at least one development partition is required")
    records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    partition_summaries: dict[str, dict[str, Any]] = {}
    decoder = {"peak_height": 0.20, "peak_prominence": 0.03,
               "peak_distance_norm": 0.06}
    for partition in partitions:
        name = str(partition["name"])
        root = Path(partition["root"])
        manifest = Path(partition["manifest"])
        role = str(partition["role"])
        cache_path = Path(partition["probability_cache"])
        expected_stems = read_manifest_stems(manifest, role)
        with np.load(cache_path) as cache:
            probabilities = np.asarray(cache["probabilities"], dtype=np.float32)
            cached_stems = [str(value) for value in cache["stems"].tolist()]
        if len(cached_stems) != len(set(cached_stems)):
            raise ValueError(f"duplicate stems in probability cache: {cache_path}")
        if set(cached_stems) != set(expected_stems):
            raise ValueError(f"cache/manifest stem mismatch: {name}")
        probability_by_stem = dict(zip(cached_stems, probabilities))
        partition_records: list[dict[str, Any]] = []
        for stem in expected_stems:
            image, label = _read_pair(root / "image", root / "label", stem)
            probability = probability_by_stem[stem]
            prediction = decode_centerline_heatmap(probability, **decoder)
            reference = extract_multirow_geometry(label, label_mode=True)
            pred_measurement = multirow_coordinate_measurement(
                rows=prediction.rows, width=image.shape[1], height=image.shape[0],
                input_status=prediction.status,
            )
            ref_measurement = multirow_coordinate_measurement(
                rows=reference.rows, width=image.shape[1], height=image.shape[0],
                input_status=reference.status,
            )
            pred_sorted = tuple(sorted(prediction.rows, key=lambda row: row.far_x_norm))
            ref_sorted = tuple(sorted(reference.rows, key=lambda row: row.far_x_norm))
            matching = match_ordered_crop_rows(pred_sorted, ref_sorted)
            mapping = {pred_index: ref_index for pred_index, ref_index in matching["pairs"]}

            def boundary_indices(
                measurement: MultiRowCoordinateMeasurement,
                original_rows: Sequence[CropRowLine],
                far_sorted: Sequence[CropRowLine],
            ) -> tuple[int, int] | None:
                if measurement.corridor_left_index is None:
                    return None
                audit_sorted = tuple(sorted(
                    original_rows, key=lambda row: row.x_at(CORRIDOR_AUDIT_Y_NORM)
                ))
                left = audit_sorted[measurement.corridor_left_index]
                right = audit_sorted[measurement.corridor_right_index]
                return far_sorted.index(left), far_sorted.index(right)

            pred_boundary = boundary_indices(pred_measurement, prediction.rows, pred_sorted)
            ref_boundary = boundary_indices(ref_measurement, reference.rows, ref_sorted)
            boundary_pair_correct = bool(
                pred_boundary is not None
                and ref_boundary is not None
                and mapping.get(pred_boundary[0]) == ref_boundary[0]
                and mapping.get(pred_boundary[1]) == ref_boundary[1]
            )
            paired_corridor = (
                pred_measurement.status == "valid" and ref_measurement.status == "valid"
            )
            center_error = (
                abs(pred_measurement.corridor_center_near_x_norm
                    - ref_measurement.corridor_center_near_x_norm)
                if paired_corridor else None
            )
            heading_error = (
                abs(pred_measurement.heading_proxy_deg
                    - ref_measurement.heading_proxy_deg)
                if paired_corridor else None
            )
            vp_error = (
                float(np.linalg.norm(
                    np.asarray(pred_measurement.vanishing_point_norm)
                    - np.asarray(ref_measurement.vanishing_point_norm)
                ))
                if pred_measurement.vanishing_point_norm is not None
                and ref_measurement.vanishing_point_norm is not None
                else None
            )
            spacing_error = (
                abs(float(np.median(pred_measurement.row_spacing_norm))
                    - float(np.median(ref_measurement.row_spacing_norm)))
                if pred_measurement.row_spacing_norm and ref_measurement.row_spacing_norm
                else None
            )
            record = {
                "partition": name, "stem": stem,
                "prediction_status": pred_measurement.status,
                "reference_status": ref_measurement.status,
                "predicted_row_count": len(prediction.rows),
                "reference_row_count": len(reference.rows),
                "boundary_pair_correct": boundary_pair_correct,
                "corridor_center_near_x_norm": pred_measurement.corridor_center_near_x_norm,
                "lateral_offset_norm": pred_measurement.lateral_offset_norm,
                "lateral_offset_px": pred_measurement.lateral_offset_px,
                "corridor_heading_proxy_deg": pred_measurement.heading_proxy_deg,
                "corridor_center_error_norm": center_error,
                "corridor_heading_error_deg": heading_error,
                "prediction_vanishing_point_status": pred_measurement.vanishing_point_status,
                "prediction_vanishing_point_x_norm": (
                    pred_measurement.vanishing_point_norm[0]
                    if pred_measurement.vanishing_point_norm is not None else None
                ),
                "prediction_vanishing_point_y_norm": (
                    pred_measurement.vanishing_point_norm[1]
                    if pred_measurement.vanishing_point_norm is not None else None
                ),
                "vanishing_point_error_norm": vp_error,
                "row_spacing_median_norm": (
                    float(np.median(pred_measurement.row_spacing_norm))
                    if pred_measurement.row_spacing_norm else None
                ),
                "row_spacing_median_error_norm": spacing_error,
                "camera_ray_status": pred_measurement.camera_ray_status,
                "metric_measurement_status": pred_measurement.metric_measurement_status,
                "reason": pred_measurement.reason,
            }
            records.append(record)
            partition_records.append(record)
            samples.append({
                "record": record, "image": image, "probability": probability,
                "prediction": prediction, "reference": reference,
                "prediction_measurement": pred_measurement,
                "reference_measurement": ref_measurement,
            })
        partition_summaries[name] = _summarize_multirow_measurements(partition_records)

    summary = _summarize_multirow_measurements(records)
    gates = {
        "corridor_boundary_pair_accuracy_at_least_0_80": (
            summary["corridor_boundary_pair_accuracy"] is not None
            and summary["corridor_boundary_pair_accuracy"] >= 0.80
        ),
        "corridor_center_mae_at_most_0_05": (
            summary["corridor_center_mae_norm"] is not None
            and summary["corridor_center_mae_norm"] <= 0.05
        ),
        "corridor_heading_mae_at_most_5_deg": (
            summary["corridor_heading_mae_deg"] is not None
            and summary["corridor_heading_mae_deg"] <= 5.0
        ),
        "vanishing_point_available_at_least_0_90": (
            summary["vanishing_point_available_fraction"] >= 0.90
        ),
        "all_camera_rays_blocked_without_calibration": all(
            row["camera_ray_status"] == "blocked_no_calibration" for row in records
        ),
        "all_metric_measurements_blocked_without_ground_transform": all(
            row["metric_measurement_status"] == "blocked_no_ground_transform"
            for row in records
        ),
    }
    safety_handoff_gates = {
        "supported_valid_recall_at_least_0_80": (
            summary["supported_valid_recall"] is not None
            and summary["supported_valid_recall"] >= 0.80
        ),
        "unsafe_false_valid_rate_at_most_0_05": (
            summary["unsafe_false_valid_rate"] is not None
            and summary["unsafe_false_valid_rate"] <= 0.05
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "coordinate_metrics_multirow.csv", records)
    review_stems = _write_multirow_contact_sheet(
        output_dir / "coordinate_contact_sheet_multirow.jpg",
        samples,
        comparison_count,
    )
    result = {
        "schema_version": 2,
        "marker": "DAY64_MULTIROW_RELEARNING_COMPLETE",
        "method": "Day63 partition-matched ResNet18 OOF multirow geometry to adjacent image corridor measurement",
        "decoder": decoder,
        "partition_summaries": partition_summaries,
        "summary": summary,
        "coordinate_contract": {
            "robot_reference": "image center x=0.5",
            "corridor_boundaries": "nearest supported crop rows strictly left and right of image center",
            "central_crop_row_rule": "if a crop row occupies |x-0.5|<=0.04, no drivable corridor center is emitted",
            "lateral_offset_norm": "corridor_center_near_x_norm - 0.5; positive means desired corridor is right of image center",
            "heading_proxy_deg": "image-space corridor center direction from y=0.90 to y=0.40",
            "vanishing_point": "robust IRLS intersection of two or more detected crop-row lines in normalized image coordinates",
            "row_spacing_norm": "adjacent row separation at y=0.80; image normalized, not metres",
        },
        "day64_measurement_gates": gates,
        "day64_measurement_gate_passed": all(gates.values()),
        "day65_safety_handoff_gates": safety_handoff_gates,
        "day65_safety_handoff_gate_passed": all(safety_handoff_gates.values()),
        "camera_calibration_available": False,
        "metric_measurement_available": False,
        "physical_measurement_blockers": [
            "camera intrinsics and lens distortion coefficients are unavailable",
            "camera-to-ground extrinsics or a calibrated homography are unavailable",
            "robot footprint and traversability labels are unavailable",
        ],
        "rejected_safety_refinements": [
            "corridor-gap-ratio threshold did not reduce unsafe false-valid rate without excessive supported-corridor loss",
            "lower heatmap-threshold consistency did not materially reduce validation-development unsafe false-valid rate",
        ],
        "review_stems": review_stems,
        "day63_probabilities_retuned": False,
        "crdld_test_data_accessed": False,
        "frozen_external_accessed": False,
        "real_robot_control_established": False,
        "evidence_boundary": "OOF predictions on already-used same-source positive development partitions; not untouched confirmation or target-domain safety evidence",
    }
    (output_dir / "day64_results_multirow.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("multirow", "single-row-history"),
                        default="multirow")
    parser.add_argument("--day63-model", type=Path)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--train-probability-cache", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-probability-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "single-row-history":
        required = (args.day63_model, args.validation_root, args.validation_manifest)
        if any(value is None for value in required):
            parser.error("single-row-history requires --day63-model, --validation-root and --validation-manifest")
        run_day64_study(
            day63_model_path=args.day63_model,
            validation_root=args.validation_root,
            validation_manifest=args.validation_manifest,
            output_dir=args.output_dir,
            comparison_count=args.count)
    else:
        required = (
            args.train_root, args.train_manifest, args.train_probability_cache,
            args.validation_root, args.validation_manifest,
            args.validation_probability_cache,
        )
        if any(value is None for value in required):
            parser.error("multirow mode requires both roots, manifests and probability caches")
        run_day64_multirow_study(
            partitions=[
                {"name": "train_development_oof", "root": args.train_root,
                 "manifest": args.train_manifest, "role": "train_development",
                 "probability_cache": args.train_probability_cache},
                {"name": "reused_validation_development_oof",
                 "root": args.validation_root, "manifest": args.validation_manifest,
                 "role": "validation_development",
                 "probability_cache": args.validation_probability_cache},
            ],
            output_dir=args.output_dir,
            comparison_count=args.count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
