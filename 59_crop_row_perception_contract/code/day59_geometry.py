from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("normalized coordinates must be within [0, 1]")


@dataclass(frozen=True)
class PathErrors:
    lateral_offset_norm: float
    heading_error_deg: float


@dataclass(frozen=True)
class CropRowGeometry:
    near: NormalizedPoint
    far: NormalizedPoint

    def __post_init__(self) -> None:
        if self.far.y >= self.near.y:
            raise ValueError("crop-row far point must be above its near point")


@dataclass(frozen=True)
class CorridorGeometry:
    left: CropRowGeometry
    right: CropRowGeometry
    center_near: NormalizedPoint
    center_far: NormalizedPoint


def normalize_pixel(
    x_px: float,
    y_px: float,
    *,
    width: int,
    height: int,
) -> NormalizedPoint:
    if width < 2 or height < 2:
        raise ValueError("image width and height must each be at least 2 pixels")
    if not (0.0 <= x_px <= width - 1 and 0.0 <= y_px <= height - 1):
        raise ValueError("pixel coordinates must be inside the image")
    return NormalizedPoint(x=x_px / (width - 1), y=y_px / (height - 1))


def compute_path_errors(*, near: NormalizedPoint, far: NormalizedPoint) -> PathErrors:
    if far.y >= near.y:
        raise ValueError("far point must be above the near point in image coordinates")

    lateral_offset_norm = near.x - 0.5
    horizontal_change = far.x - near.x
    forward_change = near.y - far.y
    heading_error_deg = degrees(atan2(horizontal_change, forward_change))
    return PathErrors(
        lateral_offset_norm=lateral_offset_norm,
        heading_error_deg=heading_error_deg,
    )


def select_camera_corridor(
    rows: list[CropRowGeometry],
    *,
    camera_center_x_norm: float = 0.5,
    center_exclusion_norm: float = 0.02,
) -> CorridorGeometry:
    """Select the nearest crop rows around the camera-centre proxy.

    This defines an image-plane corridor only.  It does not establish robot
    body boundaries, safety clearance, or metric lateral distance.
    """
    if not 0.0 <= camera_center_x_norm <= 1.0:
        raise ValueError("camera center must be within [0, 1]")
    if center_exclusion_norm < 0.0:
        raise ValueError("center exclusion must be non-negative")
    if any(abs(row.near.x - camera_center_x_norm) <= center_exclusion_norm for row in rows):
        raise ValueError("crop row intersects the camera-center exclusion zone")

    ordered = sorted(rows, key=lambda row: row.near.x)
    left_rows = [row for row in ordered if row.near.x < camera_center_x_norm]
    right_rows = [row for row in ordered if row.near.x > camera_center_x_norm]
    if not left_rows or not right_rows:
        raise ValueError("reliable crop rows are required on both sides")

    left = left_rows[-1]
    right = right_rows[0]
    if left.far.y != right.far.y or left.near.y != right.near.y:
        raise ValueError("corridor boundaries must use matching evaluation rows")
    center_near = NormalizedPoint(
        x=(left.near.x + right.near.x) / 2.0,
        y=left.near.y,
    )
    center_far = NormalizedPoint(
        x=(left.far.x + right.far.x) / 2.0,
        y=left.far.y,
    )
    return CorridorGeometry(
        left=left,
        right=right,
        center_near=center_near,
        center_far=center_far,
    )
