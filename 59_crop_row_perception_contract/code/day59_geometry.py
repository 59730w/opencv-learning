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
