from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import day59_geometry  # noqa: E402

NormalizedPoint = day59_geometry.NormalizedPoint
compute_path_errors = day59_geometry.compute_path_errors
normalize_pixel = day59_geometry.normalize_pixel


def test_normalize_pixel_maps_image_corners_to_unit_square() -> None:
    assert normalize_pixel(0, 0, width=640, height=360) == NormalizedPoint(0.0, 0.0)
    assert normalize_pixel(639, 359, width=640, height=360) == NormalizedPoint(1.0, 1.0)


def test_normalize_pixel_rejects_invalid_shape_and_out_of_bounds_pixel() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        normalize_pixel(0, 0, width=1, height=360)
    with pytest.raises(ValueError, match="inside the image"):
        normalize_pixel(640, 10, width=640, height=360)


def test_centered_vertical_path_has_zero_errors() -> None:
    errors = compute_path_errors(
        near=NormalizedPoint(0.5, 0.9),
        far=NormalizedPoint(0.5, 0.5),
    )

    assert errors.lateral_offset_norm == pytest.approx(0.0)
    assert errors.heading_error_deg == pytest.approx(0.0)


def test_lateral_offset_is_positive_when_path_is_right_of_camera_center() -> None:
    errors = compute_path_errors(
        near=NormalizedPoint(0.62, 0.9),
        far=NormalizedPoint(0.62, 0.5),
    )

    assert errors.lateral_offset_norm == pytest.approx(0.12)


def test_heading_error_sign_uses_near_to_far_direction() -> None:
    right_leaning = compute_path_errors(
        near=NormalizedPoint(0.5, 0.9),
        far=NormalizedPoint(0.6, 0.5),
    )
    left_leaning = compute_path_errors(
        near=NormalizedPoint(0.5, 0.9),
        far=NormalizedPoint(0.4, 0.5),
    )

    expected = math.degrees(math.atan2(0.1, 0.4))
    assert right_leaning.heading_error_deg == pytest.approx(expected)
    assert left_leaning.heading_error_deg == pytest.approx(-expected)


def test_normalized_point_and_path_order_are_validated() -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        NormalizedPoint(-0.01, 0.5)
    with pytest.raises(ValueError, match="far point must be above"):
        compute_path_errors(
            near=NormalizedPoint(0.5, 0.5),
            far=NormalizedPoint(0.5, 0.7),
        )


def test_corridor_center_is_midpoint_of_nearest_left_and_right_crop_rows() -> None:
    assert hasattr(day59_geometry, "CropRowGeometry")
    assert hasattr(day59_geometry, "select_camera_corridor")
    CropRowGeometry = day59_geometry.CropRowGeometry
    select_camera_corridor = day59_geometry.select_camera_corridor
    rows = [
        CropRowGeometry(NormalizedPoint(0.15, 0.9), NormalizedPoint(0.35, 0.4)),
        CropRowGeometry(NormalizedPoint(0.38, 0.9), NormalizedPoint(0.46, 0.4)),
        CropRowGeometry(NormalizedPoint(0.66, 0.9), NormalizedPoint(0.55, 0.4)),
        CropRowGeometry(NormalizedPoint(0.91, 0.9), NormalizedPoint(0.66, 0.4)),
    ]

    corridor = select_camera_corridor(rows)

    assert corridor.left.near.x == pytest.approx(0.38)
    assert corridor.right.near.x == pytest.approx(0.66)
    assert corridor.center_near == NormalizedPoint(0.52, 0.9)
    assert corridor.center_far == NormalizedPoint(0.505, 0.4)


def test_corridor_requires_crop_rows_on_both_sides() -> None:
    CropRowGeometry = day59_geometry.CropRowGeometry
    select_camera_corridor = day59_geometry.select_camera_corridor
    rows = [CropRowGeometry(NormalizedPoint(0.2, 0.9), NormalizedPoint(0.4, 0.4))]

    with pytest.raises(ValueError, match="both sides"):
        select_camera_corridor(rows)


def test_crop_row_on_camera_center_is_not_treated_as_drivable_center() -> None:
    CropRowGeometry = day59_geometry.CropRowGeometry
    select_camera_corridor = day59_geometry.select_camera_corridor
    rows = [
        CropRowGeometry(NormalizedPoint(0.3, 0.9), NormalizedPoint(0.43, 0.4)),
        CropRowGeometry(NormalizedPoint(0.5, 0.9), NormalizedPoint(0.5, 0.4)),
        CropRowGeometry(NormalizedPoint(0.7, 0.9), NormalizedPoint(0.57, 0.4)),
    ]

    with pytest.raises(ValueError, match="crop row intersects"):
        select_camera_corridor(rows)
