from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pytest


DAY63_CODE = Path(__file__).resolve().parents[2] / "63_crop_row_geometry_extraction" / "code"
DAY64_CODE = Path(__file__).resolve().parents[1] / "code"
for dependency in (DAY63_CODE, DAY64_CODE):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from day63_crop_row_geometry import (  # noqa: E402
    extract_multiscale_geometry_features,
    fit_extra_trees_geometry,
)
from day64_camera_measurement import (  # noqa: E402
    estimate_vanishing_point,
    ground_plane_measurement,
    image_coordinate_measurement,
    normalized_to_pixel,
    project_pixel_to_camera_ray,
    run_day64_study,
)


def test_normalized_to_pixel_maps_declared_corners() -> None:
    assert normalized_to_pixel((0.0, 0.0), width=321, height=241) == (0.0, 0.0)
    assert normalized_to_pixel((1.0, 1.0), width=321, height=241) == (320.0, 240.0)


def test_normalized_to_pixel_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        normalized_to_pixel((1.1, 0.5), width=320, height=240)
    with pytest.raises(ValueError):
        normalized_to_pixel((0.5, 0.5), width=1, height=240)


def test_image_offset_sign_and_vertical_heading() -> None:
    center = image_coordinate_measurement(
        near_x_norm=0.5, far_x_norm=0.5, width=640, height=480)
    right = image_coordinate_measurement(
        near_x_norm=0.62, far_x_norm=0.62, width=640, height=480)
    left = image_coordinate_measurement(
        near_x_norm=0.38, far_x_norm=0.38, width=640, height=480)
    assert center.lateral_offset_norm == pytest.approx(0.0)
    assert center.heading_proxy_deg == pytest.approx(0.0)
    assert right.lateral_offset_norm == pytest.approx(0.12)
    assert left.lateral_offset_norm == pytest.approx(-0.12)


def test_heading_sign_uses_near_to_far_direction() -> None:
    right_at_horizon = image_coordinate_measurement(
        near_x_norm=0.50, far_x_norm=0.60, width=640, height=480)
    left_at_horizon = image_coordinate_measurement(
        near_x_norm=0.50, far_x_norm=0.40, width=640, height=480)
    assert right_at_horizon.heading_proxy_deg > 0
    assert left_at_horizon.heading_proxy_deg < 0


def test_normalized_measurement_is_resolution_invariant() -> None:
    small = image_coordinate_measurement(
        near_x_norm=0.60, far_x_norm=0.52, width=320, height=240)
    large = image_coordinate_measurement(
        near_x_norm=0.60, far_x_norm=0.52, width=640, height=480)
    assert small.lateral_offset_norm == large.lateral_offset_norm
    assert small.heading_proxy_deg == large.heading_proxy_deg
    assert large.lateral_offset_px > small.lateral_offset_px


def test_day63_state_is_propagated_without_promotion() -> None:
    degraded = image_coordinate_measurement(
        near_x_norm=0.55, far_x_norm=0.50, width=640, height=480,
        input_status="degraded", confidence=0.4, uncertainty=0.09)
    rejected = image_coordinate_measurement(
        near_x_norm=None, far_x_norm=None, width=640, height=480,
        input_status="reject", confidence=0.0, uncertainty=None)
    assert degraded.status == "degraded"
    assert rejected.status == "reject"
    assert rejected.lateral_offset_norm is None


def test_horizon_intercept_is_explicitly_only_a_line_proxy() -> None:
    measurement = image_coordinate_measurement(
        near_x_norm=0.60, far_x_norm=0.50, width=640, height=480,
        horizon_y_norm=0.25)
    assert measurement.horizon_intercept_norm == pytest.approx(0.47)
    assert measurement.vanishing_point_status == "unavailable_single_line"


def test_vanishing_point_requires_two_lines() -> None:
    result = estimate_vanishing_point([((0.4, 0.9), (0.5, 0.4))])
    assert result.status == "unavailable"
    assert result.reason == "at_least_two_lines_required"


def test_vanishing_point_recovers_known_intersection() -> None:
    lines = [
        ((0.20, 1.0), (0.50, 0.20)),
        ((0.80, 1.0), (0.50, 0.20)),
        ((0.35, 1.0), (0.50, 0.20)),
    ]
    result = estimate_vanishing_point(lines)
    assert result.status == "available"
    assert result.point_norm == pytest.approx((0.50, 0.20), abs=1e-6)


def test_vanishing_point_rejects_parallel_lines() -> None:
    result = estimate_vanishing_point([
        ((0.20, 1.0), (0.30, 0.2)),
        ((0.40, 1.0), (0.50, 0.2)),
    ])
    assert result.status == "unavailable"
    assert result.reason == "lines_are_parallel_or_ill_conditioned"


def test_camera_ray_is_blocked_without_intrinsics() -> None:
    result = project_pixel_to_camera_ray((320.0, 240.0), camera_matrix=None)
    assert result.status == "blocked_no_calibration"
    assert result.ray is None


def test_camera_ray_uses_inverse_intrinsics_and_unit_norm() -> None:
    matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    center = project_pixel_to_camera_ray((320.0, 240.0), camera_matrix=matrix)
    right = project_pixel_to_camera_ray((420.0, 240.0), camera_matrix=matrix)
    assert center.ray == pytest.approx((0.0, 0.0, 1.0))
    assert np.linalg.norm(right.ray) == pytest.approx(1.0)
    assert right.ray[0] > 0


def test_metric_measurement_is_blocked_without_ground_transform() -> None:
    result = ground_plane_measurement(
        near_pixel=(320.0, 430.0), far_pixel=(320.0, 190.0),
        image_to_ground=None)
    assert result.status == "blocked_no_ground_transform"
    assert result.lateral_offset_m is None


def test_ground_transform_returns_metric_contract_for_synthetic_fixture() -> None:
    image_to_ground = np.eye(3, dtype=np.float64)
    result = ground_plane_measurement(
        near_pixel=(0.2, 1.0), far_pixel=(0.5, 3.0),
        image_to_ground=image_to_ground, robot_center_x_m=0.0)
    assert result.status == "available_synthetic_or_calibrated"
    assert result.lateral_offset_m == pytest.approx(0.2)
    assert result.heading_deg == pytest.approx(np.degrees(np.arctan2(0.3, 2.0)))


def _synthetic_row(height: int, width: int, near: float, far: float) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    near_point = (round(near * (width - 1)), round(0.90 * (height - 1)))
    far_point = (round(far * (width - 1)), round(0.40 * (height - 1)))
    cv2.line(mask, near_point, far_point, 255, 13)
    return mask


def test_tiny_day64_study_writes_results_without_physical_claims(tmp_path: Path) -> None:
    root = tmp_path / "validation"
    (root / "image").mkdir(parents=True)
    (root / "label").mkdir()
    records = []
    train_features = []
    train_targets = []
    for index, near in enumerate(np.linspace(0.44, 0.56, 12)):
        mask = _synthetic_row(120, 160, float(near), 0.50)
        train_features.append(extract_multiscale_geometry_features(mask))
        train_targets.append([near, 0.50])
    model = fit_extra_trees_geometry(
        np.asarray(train_features), np.asarray(train_targets),
        n_estimators=12, max_depth=6, min_samples_leaf=1, max_features=1.0)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"estimator": model.estimator, "feature_length": 320,
                 "selected_config": {"name": "tiny"}}, model_path)
    for index, near in enumerate((0.46, 0.50, 0.54)):
        label = _synthetic_row(120, 160, near, 0.50)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[label > 0] = (0, 180, 0)
        cv2.imwrite(str(root / "image" / f"{index}.jpg"), image)
        cv2.imwrite(str(root / "label" / f"{index}.jpg"), label)
        records.append(json.dumps({"item_id": str(index), "role": "validation_development"}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(records), encoding="utf-8")
    output = tmp_path / "output"
    result = run_day64_study(
        day63_model_path=model_path, validation_root=root,
        validation_manifest=manifest, output_dir=output, comparison_count=3)
    assert result["marker"] == "DAY64_LESSON_COMPLETE"
    assert result["day63_model_retuned"] is False
    assert result["camera_calibration_available"] is False
    assert result["metric_measurement_available"] is False
    assert result["vanishing_point_on_real_outputs"] == "BLOCKED_SINGLE_LINE"
    assert (output / "day64_results.json").is_file()
    assert (output / "coordinate_metrics.csv").is_file()
    assert (output / "coordinate_contact_sheet.jpg").is_file()
