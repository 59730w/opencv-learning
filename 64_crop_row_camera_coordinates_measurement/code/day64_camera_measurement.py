"""Day64 image coordinates and measurement boundaries for crop-row geometry.

The frozen Day63 model yields one central image line.  This module converts it
to image-plane offset and direction while refusing unsupported vanishing-point,
camera-ray, metric-distance, or robot-control claims.
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
    ExtraTreesGeometryModel,
    FAR_Y_NORM,
    NEAR_Y_NORM,
    _frozen_day62_mask,
    _read_pair,
    central_label_geometry,
    evaluate_prediction,
    extract_multiscale_geometry_features,
    geometry_prediction_from_regression,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day63-model", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    run_day64_study(
        day63_model_path=args.day63_model,
        validation_root=args.validation_root,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        comparison_count=args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
