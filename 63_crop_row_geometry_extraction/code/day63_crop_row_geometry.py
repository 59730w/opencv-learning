"""Day63 single-row baselines and multi-row geometry from frozen Day62 masks.

The current revision detects all evaluable CRDLD centerlines, orders them, and
derives an image-space corridor candidate.  The historical central-row methods
remain as baselines.  Results are development evidence, not robot-control or
external-generalization evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from scipy.signal import find_peaks
import torch
from torch import nn
from torchvision.models import resnet18


DAY61_CODE_DIR = Path(__file__).resolve().parents[2] / "61_crop_row_color_illumination" / "code"
DAY62_CODE_DIR = Path(__file__).resolve().parents[2] / "62_crop_row_morphology_regions" / "code"
for dependency in (DAY61_CODE_DIR, DAY62_CODE_DIR):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from day61_color_illumination import read_manifest_stems, segment_grayworld_hsv  # noqa: E402
from day62_morphology_regions import clean_candidate_mask  # noqa: E402


DAY62_FROZEN_CONFIG: dict[str, Any] = {
    "name": "directional_close5x7_perspective40",
    "order": "open_close",
    "open_kernel": 3,
    "close_kernel": [5, 7],
    "min_area_fraction": 0.0002,
    "perspective_top_scale": 0.4,
    "perspective_exponent": 1.0,
}

NEAR_Y_NORM = 0.90
FAR_Y_NORM = 0.40
POSITION_THRESHOLD = 0.05
HEADING_THRESHOLD_DEG = 5.0

@dataclass(frozen=True)
class GeometryConfig:
    name: str
    top_y_norm: float = 0.25
    blur_sigma_x: float = 7.0
    center_penalty: float = 1.5
    angle_penalty: float = 0.20
    near_step_norm: float = 0.015
    far_step_norm: float = 0.0125
    support_samples: int = 36


GEOMETRY_CANDIDATES = [
    GeometryConfig(
        name="support_sigma5_center15_angle20",
        blur_sigma_x=5.0,
        center_penalty=1.5,
        angle_penalty=0.20,
    ),
    GeometryConfig(
        name="support_sigma7_center10_angle20",
        blur_sigma_x=7.0,
        center_penalty=1.0,
        angle_penalty=0.20,
    ),
    GeometryConfig(
        name="support_sigma7_center15_angle20",
        blur_sigma_x=7.0,
        center_penalty=1.5,
        angle_penalty=0.20,
    ),
    GeometryConfig(
        name="support_sigma9_center10_angle20",
        blur_sigma_x=9.0,
        center_penalty=1.0,
        angle_penalty=0.20,
    ),
]

V2_MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "extra_depth8_leaf2",
        "n_estimators": 240,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": 0.7,
    },
    {
        "name": "extra_depth12_leaf2",
        "n_estimators": 240,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": 0.7,
    },
    {
        "name": "extra_full_leaf2",
        "n_estimators": 240,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.7,
    },
    {
        "name": "extra_depth12_leaf5",
        "n_estimators": 240,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
]


@dataclass(frozen=True)
class GeometryPrediction:
    method: str
    status: str
    near_x_norm: float | None
    far_x_norm: float | None
    confidence: float
    mean_support: float
    p20_support: float
    heading_deg: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtraTreesGeometryModel:
    estimator: ExtraTreesRegressor

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        tree_predictions = np.asarray(
            [tree.predict(matrix) for tree in self.estimator.estimators_],
            dtype=np.float64,
        )
        mean = np.clip(tree_predictions.mean(axis=0), 0.0, 1.0)
        endpoint_std = tree_predictions.std(axis=0)
        uncertainty = np.sqrt(np.mean(endpoint_std**2, axis=1))
        return mean, uncertainty


@dataclass(frozen=True)
class CropRowLine:
    """One image-plane crop-row line at the frozen far/near evaluation rows."""

    far_x_norm: float
    near_x_norm: float
    confidence: float
    support_band_count: int

    @property
    def heading_deg(self) -> float:
        return _heading_deg(self.near_x_norm, self.far_x_norm)

    def x_at(self, y_norm: float) -> float:
        fraction = (y_norm - FAR_Y_NORM) / (NEAR_Y_NORM - FAR_Y_NORM)
        return self.far_x_norm + fraction * (self.near_x_norm - self.far_x_norm)


@dataclass(frozen=True)
class MultiRowPrediction:
    method: str
    status: str
    rows: tuple[CropRowLine, ...]
    vanishing_point_norm: tuple[float, float] | None
    corridor_left_index: int | None
    corridor_right_index: int | None
    corridor_center: CropRowLine | None
    confidence: float
    reason: str


MULTIROW_BAND_Y_NORMS = tuple(float(value) for value in np.linspace(0.22, 0.82, 13))
MULTIROW_ANCHOR_Y_NORM = FAR_Y_NORM
CORRIDOR_AUDIT_Y_NORM = 0.80


class _RowConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TinyRowUNet(nn.Module):
    """Small heatmap model for same-source crop-row centerline learning."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder1 = _RowConvBlock(4, base_channels)
        self.encoder2 = _RowConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _RowConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.decoder2 = _RowConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.decoder1 = _RowConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level1 = self.encoder1(inputs)
        level2 = self.encoder2(nn.functional.max_pool2d(level1, 2))
        bottleneck = self.bottleneck(nn.functional.max_pool2d(level2, 2))
        decoded2 = self.decoder2(torch.cat((self.up2(bottleneck), level2), dim=1))
        decoded1 = self.decoder1(torch.cat((self.up1(decoded2), level1), dim=1))
        return self.output(decoded1)


class _RowUpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = _RowConvBlock(input_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        resized = nn.functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.layers(torch.cat((resized, skip), dim=1))


class ResNet18RowUNet(nn.Module):
    """Four-channel U-Net decoder over a locally supplied ResNet18 encoder."""

    def __init__(self, pretrained_state: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        if pretrained_state is not None:
            encoder.load_state_dict(pretrained_state)
        original_first = encoder.conv1
        self.conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_first.weight
            self.conv1.weight[:, 3:4] = original_first.weight.mean(dim=1, keepdim=True)
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up3 = _RowUpBlock(512, 256, 256)
        self.up2 = _RowUpBlock(256, 128, 128)
        self.up1 = _RowUpBlock(128, 64, 64)
        self.up0 = _RowUpBlock(64, 64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level0 = self.relu(self.bn1(self.conv1(inputs)))
        level1 = self.layer1(self.pool(level0))
        level2 = self.layer2(level1)
        level3 = self.layer3(level2)
        level4 = self.layer4(level3)
        decoded = self.up3(level4, level3)
        decoded = self.up2(decoded, level2)
        decoded = self.up1(decoded, level1)
        decoded = self.up0(decoded, level0)
        return nn.functional.interpolate(
            self.output(decoded),
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def prepare_centerline_tensor(
    image: np.ndarray,
    day62_mask: np.ndarray,
    label: np.ndarray,
    *,
    resolution: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Build RGB + frozen-Day62 input and a slightly widened binary target."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image")
    _validate_binary_mask(day62_mask)
    if label.ndim != 2:
        raise ValueError("label must be two-dimensional")
    if resolution < 32 or resolution % 4:
        raise ValueError("resolution must be a multiple of four and at least 32")
    rgb = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        (resolution, resolution),
        interpolation=cv2.INTER_AREA,
    )
    mask = cv2.resize(
        day62_mask,
        (resolution, resolution),
        interpolation=cv2.INTER_NEAREST,
    )
    target = cv2.resize(
        label,
        (resolution, resolution),
        interpolation=cv2.INTER_NEAREST,
    )
    target = cv2.dilate((target > 80).astype(np.uint8), np.ones((3, 3), np.uint8))
    features = np.concatenate((rgb, mask[:, :, None]), axis=2).transpose(2, 0, 1)
    return features.astype(np.uint8), target[None].astype(np.uint8)


def split_crossfit_folds(
    stems: list[str], *, fold_count: int = 3
) -> list[list[str]]:
    """Deterministically cover development items with disjoint OOF folds."""
    if fold_count < 2 or fold_count > len(stems):
        raise ValueError("fold_count must be between two and the sample count")
    ordered = sorted(stems, key=lambda stem: hashlib.sha256(stem.encode()).hexdigest())
    return [ordered[index::fold_count] for index in range(fold_count)]


def decode_centerline_heatmap(
    probability: np.ndarray,
    *,
    peak_height: float = 0.20,
    peak_prominence: float = 0.05,
    peak_distance_norm: float = 0.025,
) -> MultiRowPrediction:
    """Use anchor peaks for row count and the 2-D heatmap for row direction."""
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("probability must be a finite two-dimensional array")
    height, width = probability.shape
    anchor_y = round(FAR_Y_NORM * (height - 1))
    anchor_profile = probability[
        max(0, anchor_y - 2) : min(height, anchor_y + 3)
    ].mean(axis=0)
    peak_indices, properties = find_peaks(
        anchor_profile,
        height=peak_height,
        prominence=peak_prominence,
        distance=max(2, round(peak_distance_norm * width)),
    )
    binary = np.where(probability >= peak_height, 255, 0).astype(np.uint8)
    raw = extract_multirow_geometry(binary, label_mode=True)
    available = list(raw.rows)
    rows: list[CropRowLine] = []
    for peak_index, strength in zip(peak_indices, properties["peak_heights"]):
        far_x = float(peak_index / max(1, width - 1))
        if available:
            nearest_index = int(
                np.argmin([abs(row.far_x_norm - far_x) for row in available])
            )
            nearest = available[nearest_index]
        else:
            nearest = None
        if nearest is not None and abs(nearest.far_x_norm - far_x) <= 0.055:
            selected = available.pop(nearest_index)
            rows.append(
                CropRowLine(
                    far_x,
                    selected.near_x_norm + far_x - selected.far_x_norm,
                    float(strength),
                    selected.support_band_count,
                )
            )
        else:
            vp_x, vp_y = raw.vanishing_point_norm or (0.5, 0.05)
            scale = (NEAR_Y_NORM - vp_y) / max(0.08, FAR_Y_NORM - vp_y)
            near_x = vp_x + (far_x - vp_x) * scale
            rows.append(CropRowLine(far_x, float(near_x), float(strength), 1))
    rows.sort(key=lambda row: row.far_x_norm)
    ordered = tuple(rows)
    corridor = derive_image_corridor(ordered)
    indices = _corridor_indices(ordered)
    if not ordered:
        status, reason = "reject", "no learned anchor peaks"
    elif corridor is None:
        status, reason = "degraded", "rows found but safe image corridor is unavailable"
    else:
        status, reason = "valid", "learned rows and supported adjacent corridor available"
    return MultiRowPrediction(
        "tiny_unet_centerline_heatmap",
        status,
        ordered,
        raw.vanishing_point_norm,
        indices[0] if indices else None,
        indices[1] if indices else None,
        corridor,
        float(np.mean([row.confidence for row in ordered])) if ordered else 0.0,
        reason,
    )


def _estimate_mask_vanishing_point(mask: np.ndarray) -> tuple[float, float] | None:
    height, width = mask.shape
    edges = cv2.Canny(mask, 30, 100)
    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=max(18, round(0.05 * height)),
        minLineLength=max(20, round(0.07 * height)),
        maxLineGap=max(12, round(0.05 * height)),
    )
    if segments is None:
        return None
    lines: list[tuple[float, float, float]] = []
    for x1, y1, x2, y2 in segments[:, 0]:
        if abs(y2 - y1) < 0.08 * height:
            continue
        slope = (x2 - x1) / (y2 - y1)
        if abs(slope) > 1.8:
            continue
        intercept = x1 - slope * y1
        lines.append((float(slope), float(intercept), float(np.hypot(x2 - x1, y2 - y1))))
    lines = sorted(lines, key=lambda line: line[2], reverse=True)[:80]

    intersections: list[tuple[float, float]] = []
    weights: list[float] = []
    for index, (slope_a, intercept_a, length_a) in enumerate(lines):
        for slope_b, intercept_b, length_b in lines[index + 1 :]:
            if abs(slope_a - slope_b) < 0.12:
                continue
            y_px = (intercept_b - intercept_a) / (slope_a - slope_b)
            x_px = slope_a * y_px + intercept_a
            point = (x_px / (width - 1), y_px / (height - 1))
            if 0.15 <= point[0] <= 0.85 and -0.30 <= point[1] <= 0.45:
                intersections.append(point)
                weights.append(min(length_a, length_b))
    if not intersections:
        return None

    points = np.asarray(intersections, dtype=np.float64)
    base_weights = np.asarray(weights, dtype=np.float64)
    estimate = np.average(points, axis=0, weights=base_weights)
    for _ in range(20):
        distances = np.linalg.norm(points - estimate, axis=1)
        robust_weights = base_weights / np.maximum(distances, 0.02)
        updated = np.average(points, axis=0, weights=robust_weights)
        if np.linalg.norm(updated - estimate) < 1e-4:
            estimate = updated
            break
        estimate = updated
    return float(estimate[0]), float(estimate[1])


def _band_peak_points(
    mask: np.ndarray, *, label_mode: bool
) -> list[tuple[float, float, float]]:
    height, width = mask.shape
    points: list[tuple[float, float, float]] = []
    for y_norm in MULTIROW_BAND_Y_NORMS:
        y_px = round(y_norm * (height - 1))
        half_height = 3 if label_mode else max(3, round(0.01 * height))
        band = (mask[max(0, y_px - half_height) : min(height, y_px + half_height + 1)] > 0).mean(axis=0).astype(np.float32)
        if label_mode:
            active = band >= 0.12
            changes = np.diff(np.r_[False, active, False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            peaks = [(start + end - 1) / 2.0 for start, end in zip(starts, ends)]
            strengths = [1.0] * len(peaks)
        else:
            smoothed = cv2.GaussianBlur(
                band[None, :], (0, 0), sigmaX=max(3.0, 0.01 * width)
            )[0]
            peak_indices, properties = find_peaks(
                smoothed,
                distance=max(8, round(0.055 * width)),
                prominence=0.02,
                height=0.035,
            )
            peaks = peak_indices.tolist()
            strengths = properties["peak_heights"].tolist()
        for peak, strength in zip(peaks, strengths):
            points.append((float(peak / (width - 1)), y_norm, float(strength)))
    return points


def _cluster_band_points(
    points: list[tuple[float, float, float]],
    vanishing_point: tuple[float, float],
    *,
    label_mode: bool,
) -> tuple[CropRowLine, ...]:
    vp_x, vp_y = vanishing_point
    projected: list[tuple[float, float, float, float]] = []
    for x_norm, y_norm, strength in points:
        if abs(y_norm - vp_y) < 0.04:
            continue
        anchor_x = vp_x + (x_norm - vp_x) * (
            (MULTIROW_ANCHOR_Y_NORM - vp_y) / (y_norm - vp_y)
        )
        if -0.12 <= anchor_x <= 1.12:
            projected.append((anchor_x, x_norm, y_norm, strength))
    projected.sort(key=lambda item: item[0])

    tolerance = 0.026 if label_mode else 0.050
    clusters: list[list[tuple[float, float, float, float]]] = []
    for item in projected:
        candidates = [
            (abs(item[0] - float(np.median([point[0] for point in cluster]))), cluster)
            for cluster in clusters
        ]
        candidates = [candidate for candidate in candidates if candidate[0] <= tolerance]
        if candidates:
            min(candidates, key=lambda candidate: candidate[0])[1].append(item)
        else:
            clusters.append([item])

    minimum_bands = 2 if label_mode else 3
    rows: list[CropRowLine] = []
    for cluster in clusters:
        by_band: dict[float, tuple[float, float, float, float]] = {}
        for item in cluster:
            current = by_band.get(item[2])
            if current is None or item[3] > current[3]:
                by_band[item[2]] = item
        observations = list(by_band.values())
        if len(observations) < minimum_bands:
            continue
        coordinates = np.asarray([(item[2], item[1]) for item in observations], dtype=np.float64)
        keep = np.ones(len(coordinates), dtype=bool)
        residual_limit = 0.018 if label_mode else 0.045
        for _ in range(3):
            if np.count_nonzero(keep) < minimum_bands:
                break
            slope, intercept = np.polyfit(coordinates[keep, 0], coordinates[keep, 1], 1)
            residuals = np.abs(coordinates[:, 1] - (slope * coordinates[:, 0] + intercept))
            keep = residuals <= residual_limit
        if np.count_nonzero(keep) < minimum_bands:
            continue
        slope, intercept = np.polyfit(coordinates[keep, 0], coordinates[keep, 1], 1)
        far_x = float(slope * FAR_Y_NORM + intercept)
        near_x = float(slope * NEAR_Y_NORM + intercept)
        if not -0.03 <= far_x <= 1.03:
            continue
        band_fraction = np.count_nonzero(keep) / len(MULTIROW_BAND_Y_NORMS)
        mean_strength = float(np.mean([item[3] for item, accepted in zip(observations, keep) if accepted]))
        confidence = float(np.clip(0.65 * band_fraction + 0.35 * mean_strength, 0.0, 1.0))
        rows.append(CropRowLine(far_x, near_x, confidence, int(np.count_nonzero(keep))))

    rows.sort(key=lambda row: row.far_x_norm)
    merged: list[CropRowLine] = []
    minimum_separation = 0.045 if label_mode else 0.065
    for row in rows:
        if merged and abs(row.far_x_norm - merged[-1].far_x_norm) < minimum_separation:
            if row.confidence > merged[-1].confidence:
                merged[-1] = row
        else:
            merged.append(row)
    return tuple(merged)


def regularize_crop_row_lattice(
    candidates: tuple[CropRowLine, ...],
    vanishing_point: tuple[float, float],
) -> tuple[CropRowLine, ...]:
    """Fit one ordered, near-regular row family to local line candidates."""
    if len(candidates) < 3:
        return candidates
    positions = np.asarray([row.far_x_norm for row in candidates], dtype=np.float64)
    best: tuple[float, float, float, np.ndarray, list[int | None]] | None = None
    for spacing in np.arange(0.14, 0.281, 0.005):
        for phase in np.arange(-spacing, spacing + 1e-9, 0.005):
            first = math.ceil((max(0.0, positions.min() - 0.03) - phase) / spacing)
            last = math.floor((min(1.0, positions.max() + 0.03) - phase) / spacing)
            if last - first + 1 < 3:
                continue
            grid = phase + np.arange(first, last + 1) * spacing
            selected: list[int | None] = []
            support_sum = 0.0
            used: set[int] = set()
            for grid_x in grid:
                distances = np.abs(positions - grid_x)
                order = np.argsort(distances)
                index = next((int(item) for item in order if int(item) not in used), None)
                if index is not None and distances[index] <= 0.055:
                    selected.append(index)
                    used.add(index)
                    support_sum += candidates[index].confidence * math.exp(
                        -float((distances[index] / 0.035) ** 2)
                    )
                else:
                    selected.append(None)
            if len(used) < 3:
                continue
            score = support_sum - 0.35 * (len(candidates) - len(used)) - 0.08 * len(grid)
            proposal = (score, float(spacing), float(phase), grid, selected)
            if best is None or proposal[0] > best[0]:
                best = proposal
    if best is None:
        return candidates

    _, _, _, grid, selected = best
    used_indices = [index for index in selected if index is not None]
    observed_far = np.asarray([candidates[index].far_x_norm for index in used_indices])
    observed_near = np.asarray([candidates[index].near_x_norm for index in used_indices])
    if len(used_indices) >= 2:
        near_slope, near_intercept = np.polyfit(observed_far, observed_near, 1)
    else:
        vp_x, vp_y = vanishing_point
        denominator = FAR_Y_NORM - vp_y
        if abs(denominator) < 0.08:
            return candidates
        near_slope = (NEAR_Y_NORM - vp_y) / denominator
        near_intercept = vp_x * (1.0 - near_slope)
    rows: list[CropRowLine] = []
    observed_confidences = [
        candidates[index].confidence for index in selected if index is not None
    ]
    inferred_confidence = 0.35 * float(np.median(observed_confidences))
    for far_x, index in zip(grid, selected):
        if index is None:
            near_x = float(near_slope * far_x + near_intercept)
            confidence = inferred_confidence
            support_bands = 0
        else:
            near_x = candidates[index].near_x_norm + (
                float(far_x) - candidates[index].far_x_norm
            )
            confidence = candidates[index].confidence
            support_bands = candidates[index].support_band_count
        rows.append(
            CropRowLine(float(far_x), float(near_x), confidence, support_bands)
        )
    return tuple(rows)


def derive_image_corridor(
    rows: tuple[CropRowLine, ...],
    *,
    center_x_norm: float = 0.5,
    center_exclusion_norm: float = 0.04,
) -> CropRowLine | None:
    positions = [(index, row.x_at(CORRIDOR_AUDIT_Y_NORM)) for index, row in enumerate(rows)]
    if any(abs(position - center_x_norm) <= center_exclusion_norm for _, position in positions):
        return None
    left = [(index, position) for index, position in positions if position < center_x_norm]
    right = [(index, position) for index, position in positions if position > center_x_norm]
    if not left or not right:
        return None
    left_index = max(left, key=lambda item: item[1])[0]
    right_index = min(right, key=lambda item: item[1])[0]
    left_row, right_row = rows[left_index], rows[right_index]
    if left_row.support_band_count < 3 or right_row.support_band_count < 3:
        return None
    return CropRowLine(
        far_x_norm=(left_row.far_x_norm + right_row.far_x_norm) / 2.0,
        near_x_norm=(left_row.near_x_norm + right_row.near_x_norm) / 2.0,
        confidence=min(left_row.confidence, right_row.confidence),
        support_band_count=min(left_row.support_band_count, right_row.support_band_count),
    )


def extract_multirow_geometry(
    mask: np.ndarray, *, label_mode: bool = False
) -> MultiRowPrediction:
    _validate_binary_mask(mask)
    if not np.any(mask):
        return MultiRowPrediction(
            "multiband_perspective_consensus", "reject", (), None, None, None, None, 0.0, "empty mask"
        )
    band_points = _band_peak_points(mask, label_mode=label_mode)
    vanishing_point = _estimate_mask_vanishing_point(mask)
    if vanishing_point is None:
        vanishing_point = (0.5, 0.05)
    if not label_mode:
        # CRDLD's fixed forward camera places the annotated convergence region
        # close to the upper image boundary.  Intersections of lower vegetation
        # blob edges can otherwise pull the Hough estimate deep into the field.
        # Blend x with the upper-band median and bound y by the train-camera
        # prior; this is an image-domain prior, not physical calibration.
        upper_x = [x for x, y, _ in band_points if y <= 0.32]
        if upper_x:
            vanishing_point = (
                0.25 * vanishing_point[0] + 0.75 * float(np.median(upper_x)),
                float(np.clip(vanishing_point[1], -0.12, 0.04)),
            )
    rows = _cluster_band_points(
        band_points,
        vanishing_point,
        label_mode=label_mode,
    )
    if not label_mode:
        rows = regularize_crop_row_lattice(rows, vanishing_point)
    corridor = derive_image_corridor(rows)
    positions = [(index, row.x_at(CORRIDOR_AUDIT_Y_NORM)) for index, row in enumerate(rows)]
    left_index = max((item for item in positions if item[1] < 0.46), key=lambda item: item[1], default=(None, 0.0))[0]
    right_index = min((item for item in positions if item[1] > 0.54), key=lambda item: item[1], default=(None, 0.0))[0]
    if not rows:
        status, reason = "reject", "no stable multi-band row hypotheses"
    elif corridor is None:
        status, reason = "degraded", "crop rows found but image corridor is ambiguous or blocked"
        left_index = right_index = None
    else:
        status, reason = "valid", "ordered multi-band rows and adjacent image corridor available"
    confidence = float(np.mean([row.confidence for row in rows])) if rows else 0.0
    return MultiRowPrediction(
        "multiband_perspective_consensus",
        status,
        rows,
        vanishing_point,
        left_index,
        right_index,
        corridor,
        confidence,
        reason,
    )


def match_ordered_crop_rows(
    predicted: tuple[CropRowLine, ...],
    reference: tuple[CropRowLine, ...],
    *,
    max_position_error_norm: float = 0.06,
) -> dict[str, Any]:
    """Match row identity by ordered anchor position; score heading afterwards."""
    pred = tuple(sorted(predicted, key=lambda row: row.far_x_norm))
    ref = tuple(sorted(reference, key=lambda row: row.far_x_norm))
    n_pred, n_ref = len(pred), len(ref)
    dp = np.full((n_pred + 1, n_ref + 1), np.inf, dtype=np.float64)
    action = np.full((n_pred + 1, n_ref + 1), "", dtype=object)
    dp[0, 0] = 0.0
    for i in range(n_pred + 1):
        for j in range(n_ref + 1):
            current = dp[i, j]
            if not np.isfinite(current):
                continue
            if i < n_pred and current + 1.0 < dp[i + 1, j]:
                dp[i + 1, j], action[i + 1, j] = current + 1.0, "skip_pred"
            if j < n_ref and current + 1.0 < dp[i, j + 1]:
                dp[i, j + 1], action[i, j + 1] = current + 1.0, "skip_ref"
            if i < n_pred and j < n_ref:
                position_error = abs(pred[i].far_x_norm - ref[j].far_x_norm)
                if position_error <= max_position_error_norm:
                    cost = position_error / max_position_error_norm
                    if current + cost < dp[i + 1, j + 1]:
                        dp[i + 1, j + 1], action[i + 1, j + 1] = current + cost, "match"

    pairs: list[tuple[int, int]] = []
    i, j = n_pred, n_ref
    while i or j:
        step = action[i, j]
        if step == "match":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == "skip_pred":
            i -= 1
        elif step == "skip_ref":
            j -= 1
        else:
            break
    pairs.reverse()
    position_errors = [abs(pred[i].far_x_norm - ref[j].far_x_norm) for i, j in pairs]
    heading_errors = [abs(pred[i].heading_deg - ref[j].heading_deg) for i, j in pairs]
    return {
        "predicted_count": n_pred,
        "reference_count": n_ref,
        "matched_count": len(pairs),
        "pairs": pairs,
        "precision": len(pairs) / n_pred if n_pred else (1.0 if not n_ref else 0.0),
        "recall": len(pairs) / n_ref if n_ref else (1.0 if not n_pred else 0.0),
        "position_mae_norm": float(np.mean(position_errors)) if position_errors else None,
        "heading_mae_deg": float(np.mean(heading_errors)) if heading_errors else None,
    }


def _validate_binary_mask(mask: np.ndarray) -> None:
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("expected a two-dimensional uint8 binary mask")
    if not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("binary mask values must be 0 or 255")


def apply_normalized_roi(mask: np.ndarray, top_y_norm: float) -> np.ndarray:
    """Zero pixels above a normalized horizon while preserving image size."""
    _validate_binary_mask(mask)
    if not 0.0 <= top_y_norm < 1.0:
        raise ValueError("top_y_norm must be in [0, 1)")
    output = mask.copy()
    first_row = round(top_y_norm * (mask.shape[0] - 1))
    output[:first_row] = 0
    return output


def extract_multiscale_geometry_features(mask: np.ndarray) -> np.ndarray:
    """Encode coarse layout, horizontal-band moments and column occupancy."""
    _validate_binary_mask(mask)
    foreground = (mask > 0).astype(np.float32)
    pooled = cv2.resize(
        foreground, (20, 12), interpolation=cv2.INTER_AREA
    ).reshape(-1)
    height, width = foreground.shape
    x_norm = np.arange(width, dtype=np.float32) / max(1, width - 1)
    band_features: list[float] = []
    for indices in np.array_split(np.arange(height), 16):
        projection = foreground[indices].sum(axis=0)
        total = float(projection.sum())
        if total:
            mean_x = float(projection @ x_norm / total)
            std_x = float(
                np.sqrt(projection @ ((x_norm - mean_x) ** 2) / total)
            )
        else:
            mean_x = 0.5
            std_x = 0.0
        band_features.extend(
            [float(total / (len(indices) * width)), mean_x, std_x]
        )
    column_profile = cv2.resize(
        foreground.mean(axis=0).reshape(1, -1),
        (32, 1),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1)
    features = np.concatenate(
        [pooled, np.asarray(band_features, dtype=np.float32), column_profile]
    ).astype(np.float32)
    if features.shape != (320,):
        raise AssertionError("unexpected multiscale feature length")
    return features


def fit_extra_trees_geometry(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_estimators: int = 240,
    max_depth: int | None = 12,
    min_samples_leaf: int = 2,
    max_features: float = 0.7,
) -> ExtraTreesGeometryModel:
    matrix = np.asarray(features, dtype=np.float32)
    target_matrix = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or target_matrix.shape != (len(matrix), 2):
        raise ValueError("targets must contain near/far coordinates for every row")
    estimator = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=63,
        n_jobs=-1,
    )
    estimator.fit(matrix, target_matrix)
    return ExtraTreesGeometryModel(estimator=estimator)


def perspective_matrices(
    *,
    width: int,
    height: int,
    top_y_norm: float = 0.25,
    top_left_norm: float = 0.30,
    top_right_norm: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an illustrative trapezoid-to-rectangle homography and its inverse.

    This normalized transform is useful for learning and visualization only. It
    is not a calibrated bird's-eye view or a physical ground-plane mapping.
    """
    if width < 2 or height < 2:
        raise ValueError("width and height must each be at least two pixels")
    if not 0 <= top_y_norm < 1 or not 0 <= top_left_norm < top_right_norm <= 1:
        raise ValueError("invalid normalized perspective trapezoid")
    x_max = float(width - 1)
    y_max = float(height - 1)
    source = np.float32(
        [
            [top_left_norm * x_max, top_y_norm * y_max],
            [top_right_norm * x_max, top_y_norm * y_max],
            [x_max, y_max],
            [0.0, y_max],
        ]
    )
    destination = np.float32(
        [[0.0, 0.0], [x_max, 0.0], [x_max, y_max], [0.0, y_max]]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)
    return matrix, inverse


def line_points_norm(
    *, near_x_px: float, far_x_px: float, width: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    if width < 2:
        raise ValueError("width must be at least two pixels")
    denominator = width - 1
    return (
        (float(near_x_px / denominator), NEAR_Y_NORM),
        (float(far_x_px / denominator), FAR_Y_NORM),
    )


def _heading_deg(near_x_norm: float, far_x_norm: float) -> float:
    return float(
        math.degrees(
            math.atan2(far_x_norm - near_x_norm, NEAR_Y_NORM - FAR_Y_NORM)
        )
    )


def _rejected(method: str, reason: str) -> GeometryPrediction:
    return GeometryPrediction(
        method=method,
        status="reject",
        near_x_norm=None,
        far_x_norm=None,
        confidence=0.0,
        mean_support=0.0,
        p20_support=0.0,
        heading_deg=None,
        reason=reason,
    )


def perspective_support_geometry(
    mask: np.ndarray, config: GeometryConfig
) -> GeometryPrediction:
    """Search central converging line templates against perspective ROI support.

    The search is Hough-like but scores complete near/far line hypotheses.  The
    two regularizers encode the declared central-row target and a weak forward
    perspective prior; neither uses an annotation from the evaluated image.
    """
    _validate_binary_mask(mask)
    if config.blur_sigma_x <= 0 or config.support_samples < 8:
        raise ValueError("invalid geometry configuration")
    roi = apply_normalized_roi(mask, config.top_y_norm)
    if not np.any(roi):
        return _rejected(config.name, "empty ROI")

    height, width = roi.shape
    response = cv2.GaussianBlur(
        (roi > 0).astype(np.float32),
        (0, 0),
        sigmaX=config.blur_sigma_x,
        sigmaY=2.0,
    )
    near_y = NEAR_Y_NORM * (height - 1)
    far_y = FAR_Y_NORM * (height - 1)
    sample_y = np.linspace(far_y, near_y, config.support_samples)
    sample_rows = np.rint(sample_y).astype(np.int32)
    interpolation = (sample_y - far_y) / (near_y - far_y)
    near_norms = np.arange(0.18, 0.8201, config.near_step_norm)
    far_norms = np.arange(0.34, 0.6601, config.far_step_norm)

    best: tuple[float, float, float, float, float] | None = None
    for near_norm in near_norms:
        line_x_norm = far_norms[:, None] + (
            near_norm - far_norms[:, None]
        ) * interpolation[None, :]
        columns = np.clip(
            np.rint(line_x_norm * (width - 1)).astype(np.int32), 0, width - 1
        )
        values = response[sample_rows[None, :], columns]
        means = values.mean(axis=1)
        p20s = np.percentile(values, 20, axis=1)
        headings = np.abs(
            np.degrees(
                np.arctan2(far_norms - near_norm, NEAR_Y_NORM - FAR_Y_NORM)
            )
        )
        scores = (
            means
            + 0.35 * p20s
            - config.center_penalty * abs(near_norm - 0.5)
            - config.angle_penalty * headings / 45.0
        )
        index = int(np.argmax(scores))
        candidate = (
            float(scores[index]),
            float(near_norm),
            float(far_norms[index]),
            float(means[index]),
            float(p20s[index]),
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    _, near_norm, far_norm, mean_support, p20_support = best
    heading = _heading_deg(near_norm, far_norm)
    confidence = float(np.clip(0.75 * mean_support + 0.25 * p20_support, 0.0, 1.0))
    if mean_support < 0.08:
        return _rejected(config.name, "insufficient vegetation support")
    # Broken agricultural rows frequently make the lower-tail sample exactly
    # zero even when the complete line has strong repeated support.  Keep p20
    # as diagnostic evidence, but base the positive-only validity state on the
    # preregistered aggregate support score. Reject-aware calibration remains
    # unavailable until target-domain negative frames exist.
    in_central_half = 0.25 <= near_norm <= 0.75
    status = "valid" if confidence >= 0.12 and in_central_half else "degraded"
    return GeometryPrediction(
        method=config.name,
        status=status,
        near_x_norm=near_norm,
        far_x_norm=far_norm,
        confidence=confidence,
        mean_support=mean_support,
        p20_support=p20_support,
        heading_deg=heading,
        reason="full-line support search with central and perspective priors",
    )


def hough_geometry(mask: np.ndarray, top_y_norm: float = 0.25) -> GeometryPrediction:
    """Probabilistic-Hough learning baseline selected by central near intercept."""
    _validate_binary_mask(mask)
    roi = apply_normalized_roi(mask, top_y_norm)
    height, width = roi.shape
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 360,
        threshold=max(15, round(0.07 * height)),
        minLineLength=max(10, round(0.16 * height)),
        maxLineGap=max(5, round(0.10 * height)),
    )
    if lines is None:
        return _rejected("hough_p", "no Hough segment")
    candidates: list[tuple[float, float, float]] = []
    near_y = NEAR_Y_NORM * (height - 1)
    far_y = FAR_Y_NORM * (height - 1)
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y2 - y1) < 0.12 * height:
            continue
        dx_dy = (x2 - x1) / (y2 - y1)
        if abs(dx_dy) > 1.2:
            continue
        near_x = float(x1 + (near_y - y1) * dx_dy)
        far_x = float(x1 + (far_y - y1) * dx_dy)
        near_norm = near_x / (width - 1)
        far_norm = far_x / (width - 1)
        if not (-0.1 <= near_norm <= 1.1 and -0.2 <= far_norm <= 1.2):
            continue
        span = abs(y2 - y1) / height
        candidates.append((near_norm, far_norm, span))
    if not candidates:
        return _rejected("hough_p", "no plausible forward segment")
    near_norm, far_norm, span = min(
        candidates, key=lambda row: (abs(row[0] - 0.5), -row[2])
    )
    return GeometryPrediction(
        method="hough_p",
        status="valid" if span >= 0.25 else "degraded",
        near_x_norm=float(near_norm),
        far_x_norm=float(far_norm),
        confidence=float(np.clip(span, 0.0, 1.0)),
        mean_support=0.0,
        p20_support=0.0,
        heading_deg=_heading_deg(near_norm, far_norm),
        reason="central near intercept among plausible Hough segments",
    )


def _horizontal_segments(values: np.ndarray) -> list[float]:
    active = values > 0
    centers: list[float] = []
    start: int | None = None
    for index, value in enumerate(np.r_[active, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            centers.append((start + index - 1) / 2.0)
            start = None
    return centers


def central_label_geometry(label_mask: np.ndarray) -> GeometryPrediction:
    """Extract the annotated row nearest image centre at the near evaluation row."""
    _validate_binary_mask(label_mask)
    height, width = label_mask.shape
    rows = np.linspace(round(0.22 * (height - 1)), round(0.94 * (height - 1)), 20)
    previous_x = (width - 1) / 2.0
    points: list[tuple[float, float]] = []
    for y in np.rint(rows).astype(int)[::-1]:
        strip = label_mask[max(0, y - 3) : min(height, y + 4)]
        centers = _horizontal_segments(np.any(strip > 0, axis=0).astype(np.uint8))
        if not centers:
            continue
        x = min(centers, key=lambda value: abs(value - previous_x))
        points.append((x, float(y)))
        previous_x = 0.8 * x + 0.2 * previous_x
    if len(points) < 6:
        return _rejected("central_label_reference", "insufficient label points")
    array = np.asarray(points[::-1], dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(
        array, cv2.DIST_HUBER, 0, 0.01, 0.01
    ).reshape(-1)
    if abs(float(vy)) < 1e-8:
        return _rejected("central_label_reference", "horizontal label geometry")

    def x_at(y: float) -> float:
        return float(x0 + (y - y0) * vx / vy) / (width - 1)

    near_norm = x_at(NEAR_Y_NORM * (height - 1))
    far_norm = x_at(FAR_Y_NORM * (height - 1))
    return GeometryPrediction(
        method="central_label_reference",
        status="reference",
        near_x_norm=near_norm,
        far_x_norm=far_norm,
        confidence=1.0,
        mean_support=1.0,
        p20_support=1.0,
        heading_deg=_heading_deg(near_norm, far_norm),
        reason="annotation row nearest image centre at the near evaluation line",
    )


def evaluate_prediction(
    prediction: GeometryPrediction, reference: GeometryPrediction
) -> dict[str, Any]:
    if reference.near_x_norm is None or reference.heading_deg is None:
        raise ValueError("reference geometry must be available")
    valid = prediction.status != "reject" and prediction.near_x_norm is not None
    if not valid or prediction.heading_deg is None:
        return {
            "is_available": False,
            "bottom_position_abs_error_norm": None,
            "heading_abs_error_deg": None,
            "within_both_absolute_thresholds": False,
        }
    position_error = abs(prediction.near_x_norm - reference.near_x_norm)
    heading_error = abs(prediction.heading_deg - reference.heading_deg)
    return {
        "is_available": True,
        "bottom_position_abs_error_norm": float(position_error),
        "heading_abs_error_deg": float(heading_error),
        "within_both_absolute_thresholds": bool(
            position_error <= POSITION_THRESHOLD
            and heading_error <= HEADING_THRESHOLD_DEG
        ),
    }


def candidate_acceptance_checks(
    fold_summaries: list[dict[str, float]],
) -> dict[str, bool]:
    per_fold = [
        summary["valid_fraction"] >= 0.85
        and summary["bottom_position_mae_norm"] <= POSITION_THRESHOLD
        and summary["heading_mae_deg"] <= HEADING_THRESHOLD_DEG
        for summary in fold_summaries
    ]
    return {
        "exactly_five_disjoint_folds": len(fold_summaries) == 5,
        "all_five_folds_pass": len(per_fold) == 5 and all(per_fold),
    }


def v2_acceptance_checks(
    v1_folds: list[dict[str, float]], v2_folds: list[dict[str, float]]
) -> dict[str, bool]:
    """Require stable absolute and tail gains over the frozen Day63 v1."""
    paired = list(zip(v1_folds, v2_folds))
    return {
        "exactly_five_paired_folds": len(v1_folds) == len(v2_folds) == 5,
        "all_folds_valid_fraction_at_least_0_85": (
            len(paired) == 5 and all(new["valid_fraction"] >= 0.85 for _, new in paired)
        ),
        "all_folds_position_no_worse_by_more_than_0_005": (
            len(paired) == 5
            and all(
                new["bottom_position_mae_norm"]
                <= old["bottom_position_mae_norm"] + 0.005
                for old, new in paired
            )
        ),
        "all_folds_heading_improves_at_least_0_5_deg": (
            len(paired) == 5
            and all(
                new["heading_mae_deg"] <= old["heading_mae_deg"] - 0.5
                for old, new in paired
            )
        ),
        "all_folds_heading_p90_at_most_5_deg": (
            len(paired) == 5
            and all(new["heading_p90_deg"] <= 5.0 for _, new in paired)
        ),
        "all_folds_dual_gain_at_least_0_10": (
            len(paired) == 5
            and all(
                new["both_threshold_fraction_all"]
                >= old["both_threshold_fraction_all"] + 0.10
                for old, new in paired
            )
        ),
    }


def split_geometry_folds(
    stems: list[str], folds: int = 5
) -> list[list[str]]:
    """Build deterministic disjoint train-only folds for Day63 selection."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("stems must be non-empty and unique")
    ranked = sorted(
        stems,
        key=lambda stem: (
            hashlib.sha256(f"day63-geometry-fold:{stem}".encode()).hexdigest(),
            int(stem),
        ),
    )
    return [ranked[index::folds] for index in range(folds)]


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize coverage, means, tails, threshold success and runtime."""
    if not rows:
        raise ValueError("metric rows must not be empty")
    valid_rows = [
        row for row in rows if row["status"] == "valid" and row["is_available"]
    ]
    positions = [row["bottom_position_abs_error_norm"] for row in valid_rows]
    headings = [row["heading_abs_error_deg"] for row in valid_rows]
    runtimes = [row["runtime_ms"] for row in rows]
    if not positions or not headings:
        return {
            "count": float(len(rows)),
            "valid_fraction": 0.0,
            "available_fraction": float(
                np.mean([row["is_available"] for row in rows])
            ),
            "bottom_position_mae_norm": float("inf"),
            "heading_mae_deg": float("inf"),
            "position_p90_norm": float("inf"),
            "heading_p90_deg": float("inf"),
            "both_threshold_fraction_all": 0.0,
            "runtime_median_ms": float(np.median(runtimes)),
        }
    return {
        "count": float(len(rows)),
        "valid_fraction": float(
            np.mean([row["status"] == "valid" for row in rows])
        ),
        "available_fraction": float(
            np.mean([row["is_available"] for row in rows])
        ),
        "bottom_position_mae_norm": float(np.mean(positions)),
        "heading_mae_deg": float(np.mean(headings)),
        "position_p90_norm": float(np.percentile(positions, 90)),
        "heading_p90_deg": float(np.percentile(headings, 90)),
        "position_max_norm": float(np.max(positions)),
        "heading_max_deg": float(np.max(headings)),
        "both_threshold_fraction_all": float(
            np.mean([row["within_both_absolute_thresholds"] for row in rows])
        ),
        "runtime_median_ms": float(np.median(runtimes)),
    }


def select_train_only_candidate(
    summaries_by_name: dict[str, list[dict[str, float]]],
) -> str:
    eligible = [
        name
        for name, summaries in summaries_by_name.items()
        if all(candidate_acceptance_checks(summaries).values())
    ]
    if not eligible:
        raise ValueError("no geometry candidate passed all five train folds")

    def utility(name: str) -> tuple[float, float, str]:
        summaries = summaries_by_name[name]
        return (
            float(np.mean([row["bottom_position_mae_norm"] for row in summaries])),
            float(np.mean([row["heading_mae_deg"] for row in summaries])),
            name,
        )

    return min(eligible, key=utility)


def _camera_center_baseline(mask: np.ndarray) -> GeometryPrediction:
    _validate_binary_mask(mask)
    return GeometryPrediction(
        method="camera_center_vertical",
        status="valid",
        near_x_norm=0.5,
        far_x_norm=0.5,
        confidence=0.0,
        mean_support=0.0,
        p20_support=0.0,
        heading_deg=0.0,
        reason="no-image-information sanity baseline",
    )


def _read_pair(
    image_dir: Path, label_dir: Path, stem: str
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_dir / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    if image is None or label is None or image.shape[:2] != label.shape:
        raise ValueError(f"invalid image/label pair: {stem}")
    return image, np.where(label >= 128, 255, 0).astype(np.uint8)


def _frozen_day62_mask(image: np.ndarray) -> np.ndarray:
    raw = segment_grayworld_hsv(image)
    return clean_candidate_mask(raw, DAY62_FROZEN_CONFIG)


def _line_support(
    mask: np.ndarray, near_x_norm: float, far_x_norm: float
) -> tuple[float, float]:
    _validate_binary_mask(mask)
    height, width = mask.shape
    response = cv2.GaussianBlur(
        (mask > 0).astype(np.float32), (0, 0), sigmaX=9.0, sigmaY=2.0
    )
    y_norms = np.linspace(FAR_Y_NORM, NEAR_Y_NORM, 36)
    interpolation = (y_norms - FAR_Y_NORM) / (NEAR_Y_NORM - FAR_Y_NORM)
    x_norms = far_x_norm + (near_x_norm - far_x_norm) * interpolation
    rows = np.rint(y_norms * (height - 1)).astype(int)
    columns = np.clip(np.rint(x_norms * (width - 1)).astype(int), 0, width - 1)
    values = response[rows, columns]
    return float(values.mean()), float(np.percentile(values, 20))


def geometry_prediction_from_regression(
    mask: np.ndarray,
    *,
    endpoints: np.ndarray,
    uncertainty: float,
    method: str,
) -> GeometryPrediction:
    """Convert learned normalized endpoints into an explainable geometry state."""
    _validate_binary_mask(mask)
    endpoint_array = np.asarray(endpoints, dtype=np.float64).reshape(-1)
    if endpoint_array.shape != (2,) or uncertainty < 0:
        raise ValueError("expected two endpoints and non-negative uncertainty")
    if not np.any(mask):
        return _rejected(method, "empty frozen Day62 mask")
    near_norm, far_norm = np.clip(endpoint_array, 0.0, 1.0)
    mean_support, p20_support = _line_support(mask, near_norm, far_norm)
    uncertainty_quality = float(np.clip(1.0 - uncertainty / 0.08, 0.0, 1.0))
    support_quality = float(
        np.clip(0.75 * mean_support + 0.25 * p20_support, 0.0, 1.0)
    )
    confidence = 0.5 * uncertainty_quality + 0.5 * support_quality
    checks = {
        "ensemble_uncertainty_at_most_0_08": uncertainty <= 0.08,
        "near_intercept_inside_supported_range": 0.15 <= near_norm <= 0.85,
        "mean_line_support_at_least_0_05": mean_support >= 0.05,
    }
    status = "valid" if all(checks.values()) else "degraded"
    failed = [name for name, passed in checks.items() if not passed]
    return GeometryPrediction(
        method=method,
        status=status,
        near_x_norm=float(near_norm),
        far_x_norm=float(far_norm),
        confidence=float(confidence),
        mean_support=mean_support,
        p20_support=p20_support,
        heading_deg=_heading_deg(float(near_norm), float(far_norm)),
        reason=(
            f"Extra Trees endpoints; uncertainty={uncertainty:.6f}; "
            + ("all geometry checks passed" if not failed else "failed=" + ",".join(failed))
        ),
    )


def _method_prediction(
    method: str, mask: np.ndarray, configs: dict[str, GeometryConfig]
) -> GeometryPrediction:
    if method == "camera_center_vertical":
        return _camera_center_baseline(mask)
    if method == "hough_p":
        return hough_geometry(mask)
    return perspective_support_geometry(mask, configs[method])


def _evaluate_methods(
    image_dir: Path,
    label_dir: Path,
    stems: list[str],
    methods: list[str],
    configs: dict[str, GeometryConfig],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for stem in stems:
        image, label = _read_pair(image_dir, label_dir, stem)
        preprocess_start = time.perf_counter()
        mask = _frozen_day62_mask(image)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        reference = central_label_geometry(label)
        if reference.status != "reference":
            raise ValueError(f"reference geometry unavailable: {stem}")
        for method in methods:
            geometry_start = time.perf_counter()
            prediction = _method_prediction(method, mask, configs)
            geometry_ms = (time.perf_counter() - geometry_start) * 1000.0
            metrics = evaluate_prediction(prediction, reference)
            rows.append(
                {
                    "stem": stem,
                    "method": method,
                    "status": prediction.status,
                    "confidence": prediction.confidence,
                    "mean_support": prediction.mean_support,
                    "p20_support": prediction.p20_support,
                    "near_x_norm": prediction.near_x_norm,
                    "far_x_norm": prediction.far_x_norm,
                    "heading_deg": prediction.heading_deg,
                    "reference_near_x_norm": reference.near_x_norm,
                    "reference_far_x_norm": reference.far_x_norm,
                    "reference_heading_deg": reference.heading_deg,
                    "preprocess_ms": preprocess_ms,
                    "geometry_ms": geometry_ms,
                    "runtime_ms": preprocess_ms + geometry_ms,
                    **metrics,
                }
            )
    summaries = {
        method: summarize_metric_rows(
            [row for row in rows if row["method"] == method]
        )
        for method in methods
    }
    return summaries, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metrics table")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _prediction_overlay(
    image: np.ndarray,
    prediction: GeometryPrediction,
    reference: GeometryPrediction | None,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    def draw(geometry: GeometryPrediction, color: tuple[int, int, int], thickness: int) -> None:
        if geometry.near_x_norm is None or geometry.far_x_norm is None:
            return
        near = (
            round(geometry.near_x_norm * (width - 1)),
            round(NEAR_Y_NORM * (height - 1)),
        )
        far = (
            round(geometry.far_x_norm * (width - 1)),
            round(FAR_Y_NORM * (height - 1)),
        )
        cv2.line(output, far, near, color, thickness, cv2.LINE_AA)

    if reference is not None:
        draw(reference, (255, 0, 255), 2)
    draw(prediction, (0, 255, 255), 3)
    return output


def _select_review_stems(
    rows: list[dict[str, Any]], method: str, count: int
) -> list[str]:
    selected_rows = [
        row for row in rows if row["method"] == method and row["is_available"]
    ]
    if not selected_rows:
        return []
    by_position = sorted(
        selected_rows, key=lambda row: row["bottom_position_abs_error_norm"]
    )
    by_heading = sorted(selected_rows, key=lambda row: row["heading_abs_error_deg"])
    by_confidence = sorted(selected_rows, key=lambda row: row["confidence"])
    candidates = [
        by_position[0],
        by_position[len(by_position) // 2],
        by_position[-1],
        by_heading[-1],
        by_confidence[0],
        by_confidence[-1],
    ]
    candidates.extend(by_position)
    stems: list[str] = []
    for row in candidates:
        if row["stem"] not in stems:
            stems.append(row["stem"])
        if len(stems) == count:
            break
    return stems


def _write_contact_sheet(
    path: Path,
    image_dir: Path,
    label_dir: Path,
    stems: list[str],
    selected_config: GeometryConfig,
) -> None:
    rows: list[np.ndarray] = []
    for stem in stems:
        image, label = _read_pair(image_dir, label_dir, stem)
        mask = _frozen_day62_mask(image)
        hough = hough_geometry(mask)
        selected = perspective_support_geometry(mask, selected_config)
        reference = central_label_geometry(label)
        height, width = mask.shape
        matrix, _ = perspective_matrices(width=width, height=height)
        warped = cv2.warpPerspective(mask, matrix, (width, height))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        tiles = [
            _caption(image, f"{stem} input"),
            _caption(mask_bgr, "frozen Day62 mask"),
            _caption(warped_bgr, "uncalibrated perspective demo"),
            _caption(_prediction_overlay(image, hough, reference), "Hough: yellow / GT: magenta"),
            _caption(_prediction_overlay(image, selected, reference), "selected: yellow / GT: magenta"),
        ]
        tiles = [cv2.resize(tile, (220, 220), interpolation=cv2.INTER_AREA) for tile in tiles]
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


def _load_v2_samples(
    image_dir: Path, label_dir: Path, stems: list[str]
) -> list[dict[str, Any]]:
    """Materialize frozen masks, references and fixed-length v2 features once."""
    samples: list[dict[str, Any]] = []
    for stem in stems:
        image, label = _read_pair(image_dir, label_dir, stem)
        preprocess_start = time.perf_counter()
        mask = _frozen_day62_mask(image)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        feature_start = time.perf_counter()
        features = extract_multiscale_geometry_features(mask)
        feature_ms = (time.perf_counter() - feature_start) * 1000.0
        reference = central_label_geometry(label)
        if reference.status != "reference":
            raise ValueError(f"reference geometry unavailable: {stem}")
        samples.append({"stem": stem, "image": image, "mask": mask,
                        "features": features, "reference": reference,
                        "preprocess_ms": preprocess_ms, "feature_ms": feature_ms})
    return samples


def _sample_matrices(
    samples: list[dict[str, Any]], stems: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    by_stem = {sample["stem"]: sample for sample in samples}
    selected = [by_stem[stem] for stem in stems]
    features = np.asarray([sample["features"] for sample in selected], dtype=np.float32)
    targets = np.asarray(
        [[sample["reference"].near_x_norm, sample["reference"].far_x_norm]
         for sample in selected], dtype=np.float64)
    return features, targets


def _evaluate_regression_model(
    model: ExtraTreesGeometryModel,
    samples: list[dict[str, Any]],
    *,
    method: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    features = np.asarray([sample["features"] for sample in samples], dtype=np.float32)
    prediction_start = time.perf_counter()
    endpoints, uncertainties = model.predict_with_uncertainty(features)
    prediction_ms = (time.perf_counter() - prediction_start) * 1000.0 / len(samples)
    rows: list[dict[str, Any]] = []
    for sample, endpoint, uncertainty in zip(samples, endpoints, uncertainties):
        geometry_start = time.perf_counter()
        prediction = geometry_prediction_from_regression(
            sample["mask"], endpoints=endpoint, uncertainty=float(uncertainty), method=method)
        support_ms = (time.perf_counter() - geometry_start) * 1000.0
        metrics = evaluate_prediction(prediction, sample["reference"])
        geometry_ms = sample["feature_ms"] + prediction_ms + support_ms
        rows.append({
            "stem": sample["stem"], "method": method, "status": prediction.status,
            "confidence": prediction.confidence, "uncertainty": float(uncertainty),
            "mean_support": prediction.mean_support, "p20_support": prediction.p20_support,
            "near_x_norm": prediction.near_x_norm, "far_x_norm": prediction.far_x_norm,
            "heading_deg": prediction.heading_deg,
            "reference_near_x_norm": sample["reference"].near_x_norm,
            "reference_far_x_norm": sample["reference"].far_x_norm,
            "reference_heading_deg": sample["reference"].heading_deg,
            "preprocess_ms": sample["preprocess_ms"], "geometry_ms": geometry_ms,
            "runtime_ms": sample["preprocess_ms"] + geometry_ms, **metrics})
    return summarize_metric_rows(rows), rows


def _evaluate_v1_samples(
    samples: list[dict[str, Any]], config: GeometryConfig
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        geometry_start = time.perf_counter()
        prediction = perspective_support_geometry(sample["mask"], config)
        geometry_ms = (time.perf_counter() - geometry_start) * 1000.0
        metrics = evaluate_prediction(prediction, sample["reference"])
        rows.append({
            "stem": sample["stem"], "method": "v1_support_sigma9_center10_angle20",
            "status": prediction.status, "confidence": prediction.confidence,
            "uncertainty": "", "mean_support": prediction.mean_support,
            "p20_support": prediction.p20_support, "near_x_norm": prediction.near_x_norm,
            "far_x_norm": prediction.far_x_norm, "heading_deg": prediction.heading_deg,
            "reference_near_x_norm": sample["reference"].near_x_norm,
            "reference_far_x_norm": sample["reference"].far_x_norm,
            "reference_heading_deg": sample["reference"].heading_deg,
            "preprocess_ms": sample["preprocess_ms"], "geometry_ms": geometry_ms,
            "runtime_ms": sample["preprocess_ms"] + geometry_ms, **metrics})
    return summarize_metric_rows(rows), rows


def _write_v2_contact_sheet(
    path: Path, samples: list[dict[str, Any]], v1_config: GeometryConfig,
    model: ExtraTreesGeometryModel, review_stems: list[str],
) -> None:
    by_stem = {sample["stem"]: sample for sample in samples}
    rows: list[np.ndarray] = []
    for stem in review_stems:
        sample = by_stem[stem]
        endpoint, uncertainty = model.predict_with_uncertainty(sample["features"][None, :])
        v2_prediction = geometry_prediction_from_regression(
            sample["mask"], endpoints=endpoint[0], uncertainty=float(uncertainty[0]),
            method="day63_v2_extra_trees")
        v1_prediction = perspective_support_geometry(sample["mask"], v1_config)
        mask_bgr = cv2.cvtColor(sample["mask"], cv2.COLOR_GRAY2BGR)
        tiles = [
            _caption(sample["image"], f"{stem} input"),
            _caption(mask_bgr, "frozen Day62 mask"),
            _caption(_prediction_overlay(sample["image"], v1_prediction, sample["reference"]),
                     "v1 yellow / GT magenta"),
            _caption(_prediction_overlay(sample["image"], v2_prediction, sample["reference"]),
                     f"v2 yellow / GT; u={uncertainty[0]:.3f}"),
        ]
        tiles = [cv2.resize(tile, (260, 220), interpolation=cv2.INTER_AREA) for tile in tiles]
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


def run_day63_v2_study(
    *, train_root: Path, train_manifest: Path, validation_root: Path,
    validation_manifest: Path, output_dir: Path, comparison_count: int = 8,
    candidate_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select v2 on train folds and report reused validation as development only."""
    train_image_dir, train_label_dir = train_root / "image", train_root / "label"
    val_image_dir, val_label_dir = validation_root / "image", validation_root / "label"
    for directory in (train_image_dir, train_label_dir, val_image_dir, val_label_dir):
        if not directory.is_dir():
            raise ValueError(f"missing dataset directory: {directory}")

    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(validation_manifest, "validation_development")
    train_samples = _load_v2_samples(train_image_dir, train_label_dir, train_stems)
    validation_samples = _load_v2_samples(val_image_dir, val_label_dir, validation_stems)
    samples_by_stem = {sample["stem"]: sample for sample in train_samples}
    folds = split_geometry_folds(train_stems)
    configs = candidate_configs or V2_MODEL_CONFIGS
    v1_config = next(config for config in GEOMETRY_CANDIDATES
                     if config.name == "support_sigma9_center10_angle20")
    v1_fold_summaries: list[dict[str, float]] = []
    v2_fold_summaries: dict[str, list[dict[str, float]]] = {
        config["name"]: [] for config in configs}
    all_rows: list[dict[str, Any]] = []
    for fold_index, test_stems in enumerate(folds):
        test_set = set(test_stems)
        fit_stems = [stem for stem in train_stems if stem not in test_set]
        fold_samples = [samples_by_stem[stem] for stem in test_stems]
        v1_summary, v1_rows = _evaluate_v1_samples(fold_samples, v1_config)
        v1_fold_summaries.append(v1_summary)
        all_rows.extend({"partition": "train_cv", "fold": fold_index, **row}
                        for row in v1_rows)
        fit_features, fit_targets = _sample_matrices(train_samples, fit_stems)
        for config in configs:
            model = fit_extra_trees_geometry(
                fit_features, fit_targets, n_estimators=int(config["n_estimators"]),
                max_depth=config["max_depth"],
                min_samples_leaf=int(config["min_samples_leaf"]),
                max_features=float(config["max_features"]))
            summary, rows = _evaluate_regression_model(
                model, fold_samples, method=config["name"])
            v2_fold_summaries[config["name"]].append(summary)
            all_rows.extend({"partition": "train_cv", "fold": fold_index, **row}
                            for row in rows)

    checks = {name: v2_acceptance_checks(v1_fold_summaries, summaries)
              for name, summaries in v2_fold_summaries.items()}
    eligible = [name for name, gate in checks.items() if all(gate.values())]

    def selection_key(name: str) -> tuple[float, float, float, str]:
        summaries = v2_fold_summaries[name]
        return (-float(np.mean([row["both_threshold_fraction_all"] for row in summaries])),
                float(np.mean([row["bottom_position_mae_norm"] for row in summaries])),
                float(np.mean([row["heading_mae_deg"] for row in summaries])), name)

    selected_name = min(eligible or list(v2_fold_summaries), key=selection_key)
    selected_config = next(config for config in configs if config["name"] == selected_name)
    all_features, all_targets = _sample_matrices(train_samples, train_stems)
    selected_model = fit_extra_trees_geometry(
        all_features, all_targets, n_estimators=int(selected_config["n_estimators"]),
        max_depth=selected_config["max_depth"],
        min_samples_leaf=int(selected_config["min_samples_leaf"]),
        max_features=float(selected_config["max_features"]))
    v1_confirmation, v1_confirmation_rows = _evaluate_v1_samples(validation_samples, v1_config)
    v2_confirmation, v2_confirmation_rows = _evaluate_regression_model(
        selected_model, validation_samples, method=selected_name)
    development_gate = {
        "valid_fraction_at_least_0_85": v2_confirmation["valid_fraction"] >= 0.85,
        "position_mae_at_most_0_05": v2_confirmation["bottom_position_mae_norm"] <= 0.05,
        "heading_mae_at_most_5_deg": v2_confirmation["heading_mae_deg"] <= 5.0,
        "position_p90_at_most_0_08": v2_confirmation["position_p90_norm"] <= 0.08,
        "heading_p90_at_most_5_deg": v2_confirmation["heading_p90_deg"] <= 5.0,
        "dual_threshold_fraction_at_least_0_80": v2_confirmation["both_threshold_fraction_all"] >= 0.80,
        "runtime_median_at_most_50_ms": v2_confirmation["runtime_median_ms"] <= 50.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows.extend({"partition": "reused_validation", "fold": "", **row}
                    for row in v1_confirmation_rows + v2_confirmation_rows)
    _write_csv(output_dir / "geometry_metrics_v2.csv", all_rows)
    joblib.dump({
        "estimator": selected_model.estimator,
        "feature_contract": "20x12 pooled mask + 16x(density,mean_x,std_x) + 32-column profile",
        "feature_length": 320,
        "target_contract": ["near_x_norm_at_y0.90", "far_x_norm_at_y0.40"],
        "selected_config": selected_config,
        "day62_frozen_config": DAY62_FROZEN_CONFIG,
    }, output_dir / "day63_geometry_v2.joblib")
    review_stems = _select_review_stems(v2_confirmation_rows, selected_name, comparison_count)
    _write_v2_contact_sheet(output_dir / "geometry_contact_sheet_v2.jpg",
                            validation_samples, v1_config, selected_model, review_stems)
    result = {
        "schema_version": 2, "marker": "DAY63_V2_LESSON_COMPLETE",
        "scope": "CRDLD same-source positive development geometry only; v2 is not independent confirmation, reject-aware evaluation, physical measurement, or robot-control evidence",
        "reason_for_v2": "v1 passed mean-error gates but its dual-threshold success and tail errors were not satisfactory for the Day64 handoff",
        "day61_color_retuned": False, "day62_morphology_retuned": False,
        "day62_input_configuration": DAY62_FROZEN_CONFIG,
        "feature_contract": "320 fixed features: 20x12 pooled mask, 16 horizontal-band moments, and 32-column occupancy",
        "target_contract": "near and far normalized x coordinates at y=0.90 and y=0.40",
        "selection_data": "CRDLD train-development only",
        "selection_protocol": "five deterministic disjoint folds; v2 must improve v1 foldwise, including heading tail and dual-threshold success",
        "fold_sizes": [len(fold) for fold in folds], "v1_fold_summaries": v1_fold_summaries,
        "v2_fold_summaries": v2_fold_summaries, "candidate_checks": checks,
        "selected_before_reused_validation": selected_name,
        "selected_train_gate_passed": selected_name in eligible,
        "selected_config": selected_config,
        "reused_validation_development": {
            "v1": v1_confirmation, "v2": v2_confirmation, "checks": development_gate,
            "all_checks_passed": all(development_gate.values())},
        "review_stems": review_stems,
        "confidence_limitation": "Confidence combines ensemble agreement and frozen-mask line support; it is not a calibrated probability.",
        "untouched_confirmation_available": False,
        "same_source_internal_benchmark_accessed": False,
        "frozen_external_accessed": False, "reject_aware_evaluation_available": False,
        "generalization_status": "BLOCKED_UNTIL_REJECT_AWARE_AND_FROZEN_EXTERNAL_TESTS",
        "day64_handoff_status": "READY_FOR_DEFINITION_WORK"
        if selected_name in eligible and all(development_gate.values())
        else "BLOCKED_BY_DAY63_V2_GATE",
    }
    (output_dir / "day63_results_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_day63_study(
    *,
    train_root: Path,
    train_manifest: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Select geometry on train folds, then confirm once on reused validation."""
    train_image_dir, train_label_dir = train_root / "image", train_root / "label"
    val_image_dir, val_label_dir = validation_root / "image", validation_root / "label"
    for directory in (train_image_dir, train_label_dir, val_image_dir, val_label_dir):
        if not directory.is_dir():
            raise ValueError(f"missing dataset directory: {directory}")

    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(
        validation_manifest, "validation_development"
    )
    folds = split_geometry_folds(train_stems)
    configs = {config.name: config for config in GEOMETRY_CANDIDATES}
    candidate_names = list(configs)
    methods = ["camera_center_vertical", "hough_p", *candidate_names]

    fold_summaries_by_method: dict[str, list[dict[str, float]]] = {
        method: [] for method in methods
    }
    train_rows: list[dict[str, Any]] = []
    for fold_index, fold_stems in enumerate(folds):
        summaries, rows = _evaluate_methods(
            train_image_dir, train_label_dir, fold_stems, methods, configs
        )
        for method in methods:
            fold_summaries_by_method[method].append(summaries[method])
        train_rows.extend(
            {"partition": "train_cv", "fold": fold_index, **row} for row in rows
        )

    selection_error: str | None = None
    try:
        selected_name = select_train_only_candidate(
            {name: fold_summaries_by_method[name] for name in candidate_names}
        )
        selected_accepted = True
    except ValueError as error:
        selection_error = str(error)
        selected_name = min(
            candidate_names,
            key=lambda name: (
                np.mean(
                    [
                        row["bottom_position_mae_norm"]
                        for row in fold_summaries_by_method[name]
                    ]
                ),
                np.mean(
                    [row["heading_mae_deg"] for row in fold_summaries_by_method[name]]
                ),
            ),
        )
        selected_accepted = False

    confirmation_methods = ["camera_center_vertical", "hough_p", selected_name]
    confirmation_summaries, confirmation_rows = _evaluate_methods(
        val_image_dir,
        val_label_dir,
        validation_stems,
        confirmation_methods,
        configs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = train_rows + [
        {"partition": "reused_validation", "fold": "", **row}
        for row in confirmation_rows
    ]
    _write_csv(output_dir / "geometry_metrics.csv", all_rows)
    review_stems = _select_review_stems(
        confirmation_rows, selected_name, comparison_count
    )
    _write_contact_sheet(
        output_dir / "geometry_contact_sheet.jpg",
        val_image_dir,
        val_label_dir,
        review_stems,
        configs[selected_name],
    )

    result = {
        "schema_version": 1,
        "marker": "DAY63_LESSON_COMPLETE",
        "scope": "CRDLD same-source positive geometry development only; no reject-aware, external, physical-distance, or robot-control claim",
        "day61_color_retuned": False,
        "day62_morphology_retuned": False,
        "day62_input_configuration": DAY62_FROZEN_CONFIG,
        "selection_data": "CRDLD train-development only",
        "selection_protocol": "five deterministic disjoint folds; absolute position, heading, and valid-fraction thresholds must pass in every fold",
        "preliminary_round": {
            "preserved_output": "day63_results_preliminary_rejected.json",
            "outcome": "rejected because p20_support alone downgraded broken but repeatedly supported rows, leaving train-fold valid fractions near 0.51-0.59",
            "change_basis": "train-development folds only",
            "change": "retain p20 as a diagnostic but classify positive-only validity from aggregate support confidence >= 0.12",
            "validation_was_already_accessed": True,
        },
        "second_round": {
            "preserved_output": "day63_results_round2_train_gate_only.json",
            "outcome": "the train-fold geometry gate passed, but reused validation-development exposed high-error hypotheses at the search boundary",
            "change": "mark near intercepts outside the central half as degraded and report position/heading error on valid outputs only",
            "change_basis": "target definition plus train and already-reused validation-development evidence",
        },
        "fold_sizes": [len(fold) for fold in folds],
        "methods": {
            "camera_center_vertical": "no-image-information sanity baseline",
            "hough_p": "probabilistic Hough baseline",
            **{name: asdict(config) for name, config in configs.items()},
        },
        "fold_summaries_by_method": fold_summaries_by_method,
        "candidate_checks": {
            name: candidate_acceptance_checks(fold_summaries_by_method[name])
            for name in candidate_names
        },
        "selected_before_confirmation": selected_name,
        "selected_train_gate_passed": selected_accepted,
        "selection_error": selection_error,
        "confirmation_data": "CRDLD validation-development images reused during Day62 and Day63 development; no untouched confirmation remains",
        "confirmation_is_untouched": False,
        "status_gate_used_reused_validation_development": True,
        "confirmation_summaries": confirmation_summaries,
        "review_stems": review_stems,
        "perspective_limitation": "The displayed homography is normalized and uncalibrated; it is not a physical bird's-eye transform.",
        "confidence_limitation": "Confidence is a recorded geometry-support score, not a calibrated probability.",
        "same_source_internal_benchmark_accessed": False,
        "frozen_external_accessed": False,
        "reject_aware_evaluation_available": False,
        "generalization_status": "BLOCKED_UNTIL_REJECT_AWARE_AND_FROZEN_EXTERNAL_TESTS",
    }
    (output_dir / "day63_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _corridor_indices(rows: tuple[CropRowLine, ...]) -> tuple[int, int] | None:
    if derive_image_corridor(rows) is None:
        return None
    positions = [(index, row.x_at(CORRIDOR_AUDIT_Y_NORM)) for index, row in enumerate(rows)]
    if any(abs(position - 0.5) <= 0.04 for _, position in positions):
        return None
    left = [(index, position) for index, position in positions if position < 0.5]
    right = [(index, position) for index, position in positions if position > 0.5]
    if not left or not right:
        return None
    return max(left, key=lambda item: item[1])[0], min(right, key=lambda item: item[1])[0]


def _evaluate_multirow_split(
    root: Path, stems: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray, MultiRowPrediction, MultiRowPrediction]]]:
    records: list[dict[str, Any]] = []
    review: dict[str, tuple[np.ndarray, np.ndarray, MultiRowPrediction, MultiRowPrediction]] = {}
    for stem in stems:
        image, label = _read_pair(root / "image", root / "label", stem)
        mask = _frozen_day62_mask(image)
        reference = extract_multirow_geometry(label, label_mode=True)
        start = time.perf_counter()
        prediction = extract_multirow_geometry(mask, label_mode=False)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        matching = match_ordered_crop_rows(prediction.rows, reference.rows)
        mapping = {pred_index: ref_index for pred_index, ref_index in matching["pairs"]}
        pred_corridor_indices = _corridor_indices(prediction.rows)
        ref_corridor_indices = _corridor_indices(reference.rows)
        boundary_pair_correct = False
        corridor_center_error = None
        if pred_corridor_indices is not None and ref_corridor_indices is not None:
            boundary_pair_correct = (
                mapping.get(pred_corridor_indices[0]) == ref_corridor_indices[0]
                and mapping.get(pred_corridor_indices[1]) == ref_corridor_indices[1]
            )
            assert prediction.corridor_center is not None
            assert reference.corridor_center is not None
            corridor_center_error = abs(
                prediction.corridor_center.near_x_norm
                - reference.corridor_center.near_x_norm
            )
        record = {
            "stem": stem,
            "predicted_count": matching["predicted_count"],
            "reference_count": matching["reference_count"],
            "matched_count": matching["matched_count"],
            "precision": matching["precision"],
            "recall": matching["recall"],
            "matched_position_mae_norm": matching["position_mae_norm"],
            "matched_heading_mae_deg": matching["heading_mae_deg"],
            "prediction_status": prediction.status,
            "reference_corridor_available": ref_corridor_indices is not None,
            "prediction_corridor_available": pred_corridor_indices is not None,
            "boundary_pair_correct": boundary_pair_correct,
            "corridor_center_error_norm": corridor_center_error,
            "runtime_ms": runtime_ms,
        }
        records.append(record)
        review[stem] = (image, mask, prediction, reference)

    total_predicted = sum(record["predicted_count"] for record in records)
    total_reference = sum(record["reference_count"] for record in records)
    total_matched = sum(record["matched_count"] for record in records)
    matched_weight = max(1, total_matched)
    position_sum = sum(
        record["matched_position_mae_norm"] * record["matched_count"]
        for record in records
        if record["matched_position_mae_norm"] is not None
    )
    heading_sum = sum(
        record["matched_heading_mae_deg"] * record["matched_count"]
        for record in records
        if record["matched_heading_mae_deg"] is not None
    )
    supported = [record for record in records if record["reference_corridor_available"]]
    unsupported = [record for record in records if not record["reference_corridor_available"]]
    corridor_errors = [
        record["corridor_center_error_norm"]
        for record in supported
        if record["corridor_center_error_norm"] is not None
    ]
    summary = {
        "frame_count": len(records),
        "predicted_row_count": total_predicted,
        "reference_row_count": total_reference,
        "matched_row_count": total_matched,
        "row_detection_precision": total_matched / total_predicted if total_predicted else 0.0,
        "row_detection_recall": total_matched / total_reference if total_reference else 0.0,
        "matched_bottom_position_mae_norm": position_sum / matched_weight,
        "matched_heading_mae_deg": heading_sum / matched_weight,
        "reference_corridor_frame_count": len(supported),
        "corridor_boundary_pair_accuracy": (
            sum(record["boundary_pair_correct"] for record in supported) / len(supported)
            if supported else 0.0
        ),
        "corridor_center_mae_norm": float(np.mean(corridor_errors)) if corridor_errors else None,
        "supported_valid_recall": (
            sum(record["prediction_status"] == "valid" for record in supported) / len(supported)
            if supported else 0.0
        ),
        "unsafe_false_valid_rate": (
            sum(record["prediction_status"] == "valid" for record in unsupported) / len(unsupported)
            if unsupported else 0.0
        ),
        "runtime_median_ms": float(np.median([record["runtime_ms"] for record in records])) if records else 0.0,
    }
    summary["gates"] = {
        "row_detection_precision_at_least_0_80": summary["row_detection_precision"] >= 0.80,
        "row_detection_recall_at_least_0_80": summary["row_detection_recall"] >= 0.80,
        "matched_position_mae_at_most_0_05": summary["matched_bottom_position_mae_norm"] <= 0.05,
        "matched_heading_mae_at_most_5_deg": summary["matched_heading_mae_deg"] <= 5.0,
        "corridor_boundary_pair_accuracy_at_least_0_80": summary["corridor_boundary_pair_accuracy"] >= 0.80,
        "corridor_center_mae_at_most_0_05": summary["corridor_center_mae_norm"] is not None and summary["corridor_center_mae_norm"] <= 0.05,
        "supported_valid_recall_at_least_0_80": summary["supported_valid_recall"] >= 0.80,
        "unsafe_false_valid_rate_at_most_0_05": summary["unsafe_false_valid_rate"] <= 0.05,
        "runtime_median_at_most_50_ms": summary["runtime_median_ms"] <= 50.0,
    }
    summary["all_gates_passed"] = all(summary["gates"].values())
    return summary, records, review


def _draw_multirow_prediction(
    image: np.ndarray,
    prediction: MultiRowPrediction,
    reference: MultiRowPrediction,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    def draw_rows(rows: tuple[CropRowLine, ...], color: tuple[int, int, int], thickness: int) -> None:
        for row in rows:
            cv2.line(
                output,
                (round(row.far_x_norm * (width - 1)), round(FAR_Y_NORM * (height - 1))),
                (round(row.near_x_norm * (width - 1)), round(NEAR_Y_NORM * (height - 1))),
                color,
                thickness,
                cv2.LINE_AA,
            )

    draw_rows(reference.rows, (255, 0, 255), 3)
    draw_rows(prediction.rows, (0, 255, 255), 2)
    if prediction.corridor_center is not None:
        draw_rows((prediction.corridor_center,), (255, 255, 0), 3)
    cv2.putText(
        output,
        f"pred={len(prediction.rows)} gt={len(reference.rows)} {prediction.status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _write_multirow_contact_sheet(
    path: Path,
    review: dict[str, tuple[np.ndarray, np.ndarray, MultiRowPrediction, MultiRowPrediction]],
    records: list[dict[str, Any]],
    count: int,
) -> list[str]:
    ranked = sorted(
        records,
        key=lambda record: (record["recall"], record["precision"], -(record["matched_heading_mae_deg"] or 99.0)),
    )
    selected = ranked[: max(1, count // 2)] + ranked[-max(1, count - count // 2) :]
    seen: set[str] = set()
    rows: list[np.ndarray] = []
    stems: list[str] = []
    for record in selected:
        stem = record["stem"]
        if stem in seen:
            continue
        seen.add(stem)
        stems.append(stem)
        image, mask, prediction, reference = review[stem]
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = _draw_multirow_prediction(image, prediction, reference)
        rows.append(np.hstack((image, mask_bgr, overlay)))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))
    return stems


def run_day63_multirow_study(
    *,
    train_root: Path,
    train_manifest: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Run the scoped multi-row revision without touching frozen external data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(
        validation_manifest, "validation_development"
    )
    train_summary, train_records, _ = _evaluate_multirow_split(train_root, train_stems)
    validation_summary, validation_records, review = _evaluate_multirow_split(
        validation_root, validation_stems
    )
    _write_csv(output_dir / "geometry_metrics_multirow.csv", validation_records)
    review_stems = _write_multirow_contact_sheet(
        output_dir / "geometry_comparison_multirow.jpg",
        review,
        validation_records,
        comparison_count,
    )
    result = {
        "schema_version": 1,
        "marker": "DAY63_MULTIROW_REVISION_COMPLETE",
        "method": "multiband_perspective_consensus",
        "day61_color_retuned": False,
        "day62_morphology_retuned": False,
        "single_row_v2_preserved_as_baseline": True,
        "label_identity_scope": "derived_ordered_matching",
        "label_limitation": "CRDLD merges all centerlines in one binary JPEG mask and provides no instance IDs",
        "corridor_scope": "image proxy only when both sides exist and no crop row occupies the camera-center exclusion band",
        "train_development": train_summary,
        "reused_validation_development": validation_summary,
        "review_stems": review_stems,
        "same_source_internal_benchmark_accessed": False,
        "frozen_external_accessed": False,
        "reject_aware_evaluation_available": False,
        "real_robot_corridor_established": False,
    }
    (output_dir / "day63_results_multirow.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _load_centerline_samples(
    root: Path,
    stems: list[str],
    *,
    partition: str,
    resolution: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for stem in stems:
        image, label = _read_pair(root / "image", root / "label", stem)
        mask = _frozen_day62_mask(image)
        features, target = prepare_centerline_tensor(
            image, mask, label, resolution=resolution
        )
        samples.append(
            {
                "key": f"{partition}/{stem}",
                "stem": stem,
                "partition": partition,
                "image": image,
                "mask": mask,
                "features": features,
                "target": target,
                "reference": extract_multirow_geometry(label, label_mode=True),
            }
        )
    return samples


def _train_centerline_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: str,
    base_channels: int = 16,
) -> TinyRowUNet:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = TinyRowUNet(base_channels=base_channels).to(device)
    target_float = targets.astype(np.float32)
    positives = float(target_float.sum())
    negatives = float(target_float.size - positives)
    positive_weight = min(20.0, negatives / max(1.0, positives))
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features), torch.from_numpy(targets)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=24 if device.startswith("cuda") else 8,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    for _ in range(epochs):
        model.train()
        for inputs, labels in loader:
            inputs = inputs.to(device=device, dtype=torch.float32) / 255.0
            labels = labels.to(device=device, dtype=torch.float32)
            logits = model(inputs)
            probability = torch.sigmoid(logits)
            dice_loss = 1.0 - (
                2.0 * (probability * labels).sum(dim=(1, 2, 3)) + 1.0
            ) / (
                (probability + labels).sum(dim=(1, 2, 3)) + 1.0
            )
            loss = bce(logits, labels) + dice_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _predict_centerline_heatmaps(
    model: TinyRowUNet,
    features: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, float]:
    model.eval()
    batches: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(0, len(features), 32):
            inputs = torch.from_numpy(features[index : index + 32]).to(
                device=device, dtype=torch.float32
            ) / 255.0
            batches.append(torch.sigmoid(model(inputs)).cpu().numpy()[:, 0])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.concatenate(batches), elapsed_ms / max(1, len(features))


def _evaluate_centerline_predictions(
    samples: list[dict[str, Any]],
    probabilities: list[np.ndarray],
    runtimes_ms: list[float | None],
    *,
    peak_height: float = 0.20,
    peak_prominence: float = 0.05,
    peak_distance_norm: float = 0.025,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for sample, probability, runtime_ms in zip(samples, probabilities, runtimes_ms):
        prediction = decode_centerline_heatmap(
            probability,
            peak_height=peak_height,
            peak_prominence=peak_prominence,
            peak_distance_norm=peak_distance_norm,
        )
        reference = sample["reference"]
        matching = match_ordered_crop_rows(
            prediction.rows, reference.rows, max_position_error_norm=0.05
        )
        mapping = {left: right for left, right in matching["pairs"]}
        predicted_corridor = _corridor_indices(prediction.rows)
        reference_corridor = _corridor_indices(reference.rows)
        boundary_correct = False
        corridor_error = None
        if predicted_corridor is not None and reference_corridor is not None:
            boundary_correct = (
                mapping.get(predicted_corridor[0]) == reference_corridor[0]
                and mapping.get(predicted_corridor[1]) == reference_corridor[1]
            )
            if prediction.corridor_center is not None and reference.corridor_center is not None:
                corridor_error = abs(
                    prediction.corridor_center.near_x_norm
                    - reference.corridor_center.near_x_norm
                )
        records.append(
            {
                "key": sample["key"],
                "stem": sample["stem"],
                "partition": sample["partition"],
                "predicted_count": matching["predicted_count"],
                "reference_count": matching["reference_count"],
                "matched_count": matching["matched_count"],
                "precision": matching["precision"],
                "recall": matching["recall"],
                "matched_position_mae_norm": matching["position_mae_norm"],
                "matched_heading_mae_deg": matching["heading_mae_deg"],
                "prediction_status": prediction.status,
                "reference_corridor_available": reference_corridor is not None,
                "prediction_corridor_available": predicted_corridor is not None,
                "boundary_pair_correct": boundary_correct,
                "corridor_center_error_norm": corridor_error,
                "runtime_ms": runtime_ms,
            }
        )
        sample["centerline_prediction"] = prediction

    total_predicted = sum(row["predicted_count"] for row in records)
    total_reference = sum(row["reference_count"] for row in records)
    total_matched = sum(row["matched_count"] for row in records)
    position_sum = sum(
        (row["matched_position_mae_norm"] or 0.0) * row["matched_count"]
        for row in records
    )
    heading_sum = sum(
        (row["matched_heading_mae_deg"] or 0.0) * row["matched_count"]
        for row in records
    )
    supported = [row for row in records if row["reference_corridor_available"]]
    unsupported = [row for row in records if not row["reference_corridor_available"]]
    corridor_errors = [
        row["corridor_center_error_norm"]
        for row in supported
        if row["corridor_center_error_norm"] is not None
    ]
    summary = {
        "frame_count": len(records),
        "predicted_row_count": total_predicted,
        "reference_row_count": total_reference,
        "matched_row_count": total_matched,
        "row_detection_precision": total_matched / max(1, total_predicted),
        "row_detection_recall": total_matched / max(1, total_reference),
        "matched_position_mae_norm": position_sum / max(1, total_matched),
        "matched_heading_mae_deg": heading_sum / max(1, total_matched),
        "reference_corridor_frame_count": len(supported),
        "corridor_boundary_pair_accuracy": (
            sum(row["boundary_pair_correct"] for row in supported) / max(1, len(supported))
        ),
        "corridor_center_mae_norm": (
            float(np.mean(corridor_errors)) if corridor_errors else None
        ),
        "supported_valid_recall": (
            sum(row["prediction_status"] == "valid" for row in supported)
            / max(1, len(supported))
        ),
        "unsafe_false_valid_rate": (
            sum(row["prediction_status"] == "valid" for row in unsupported)
            / max(1, len(unsupported))
        ),
        "runtime_median_ms": (
            float(np.median([value for value in runtimes_ms if value is not None]))
            if any(value is not None for value in runtimes_ms)
            else None
        ),
    }
    summary["day63_geometry_gates"] = {
        "precision_at_least_0_80": summary["row_detection_precision"] >= 0.80,
        "recall_at_least_0_80": summary["row_detection_recall"] >= 0.80,
        "position_mae_at_most_0_05": summary["matched_position_mae_norm"] <= 0.05,
        "heading_mae_at_most_5_deg": summary["matched_heading_mae_deg"] <= 5.0,
    }
    summary["day63_geometry_gate_passed"] = all(
        summary["day63_geometry_gates"].values()
    )
    return summary, records


def run_day63_centerline_study(
    *,
    train_root: Path,
    train_manifest: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    comparison_count: int = 8,
    fold_count: int = 3,
    epochs: int = 12,
    resolution: int = 128,
) -> dict[str, Any]:
    """Cross-fit a learned multi-row heatmap on all declared development data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(
        validation_manifest, "validation_development"
    )
    samples = _load_centerline_samples(
        train_root, train_stems, partition="train_development", resolution=resolution
    ) + _load_centerline_samples(
        validation_root,
        validation_stems,
        partition="reused_validation_development",
        resolution=resolution,
    )
    by_key = {sample["key"]: sample for sample in samples}
    folds = split_crossfit_folds(list(by_key), fold_count=fold_count)
    probability_by_key: dict[str, np.ndarray] = {}
    runtime_by_key: dict[str, float] = {}
    fold_records: list[dict[str, Any]] = []
    all_features = np.asarray([sample["features"] for sample in samples], dtype=np.uint8)
    all_targets = np.asarray([sample["target"] for sample in samples], dtype=np.uint8)
    key_to_index = {sample["key"]: index for index, sample in enumerate(samples)}
    for fold_index, held_keys in enumerate(folds):
        held_set = set(held_keys)
        fit_indices = [
            index for index, sample in enumerate(samples) if sample["key"] not in held_set
        ]
        held_indices = [key_to_index[key] for key in held_keys]
        model = _train_centerline_model(
            all_features[fit_indices],
            all_targets[fit_indices],
            epochs=epochs,
            seed=6300 + fold_index,
            device=device,
        )
        probabilities, runtime_ms = _predict_centerline_heatmaps(
            model, all_features[held_indices], device=device
        )
        for index, probability in zip(held_indices, probabilities):
            key = samples[index]["key"]
            probability_by_key[key] = probability
            runtime_by_key[key] = runtime_ms
        fold_records.append(
            {
                "fold": fold_index,
                "fit_count": len(fit_indices),
                "held_out_count": len(held_indices),
            }
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    ordered_probabilities = [probability_by_key[sample["key"]] for sample in samples]
    ordered_runtimes = [runtime_by_key[sample["key"]] for sample in samples]
    overall_summary, records = _evaluate_centerline_predictions(
        samples, ordered_probabilities, ordered_runtimes
    )
    partition_summaries: dict[str, Any] = {}
    for partition in ("train_development", "reused_validation_development"):
        indices = [
            index for index, sample in enumerate(samples) if sample["partition"] == partition
        ]
        partition_summaries[partition], _ = _evaluate_centerline_predictions(
            [samples[index] for index in indices],
            [ordered_probabilities[index] for index in indices],
            [ordered_runtimes[index] for index in indices],
        )

    final_model = _train_centerline_model(
        all_features,
        all_targets,
        epochs=epochs,
        seed=6399,
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_model.cpu().state_dict(),
            "resolution": resolution,
            "base_channels": 16,
            "peak_height": 0.20,
            "peak_prominence": 0.05,
            "peak_distance_norm": 0.025,
        },
        output_dir / "day63_multirow_centerline_model.pt",
    )
    _write_csv(output_dir / "geometry_metrics_centerline_oof.csv", records)
    ranked = sorted(records, key=lambda row: (row["recall"], row["precision"]))
    review_rows = ranked[: comparison_count // 2] + ranked[-(comparison_count - comparison_count // 2) :]
    contact_rows: list[np.ndarray] = []
    review_stems: list[str] = []
    for row in review_rows:
        sample = by_key[row["key"]]
        review_stems.append(row["key"])
        overlay = _draw_multirow_prediction(
            sample["image"], sample["centerline_prediction"], sample["reference"]
        )
        mask_bgr = cv2.cvtColor(sample["mask"], cv2.COLOR_GRAY2BGR)
        contact_rows.append(np.hstack((sample["image"], mask_bgr, overlay)))
    if contact_rows:
        cv2.imwrite(
            str(output_dir / "geometry_comparison_centerline_oof.jpg"),
            np.vstack(contact_rows),
        )
    result = {
        "schema_version": 2,
        "marker": "DAY63_MULTIROW_RELEARNING_COMPLETE",
        "method": "three_fold_cross_fitted_tiny_unet_centerline_heatmap",
        "development_selection_disclosure": "decoder settings were selected after the reused validation-development split had already been accessed; OOF labels are excluded from model fitting but this is not untouched confirmation",
        "folds": fold_records,
        "overall_out_of_fold": overall_summary,
        "partition_out_of_fold": partition_summaries,
        "review_stems": review_stems,
        "day61_color_retuned": False,
        "day62_morphology_retuned": False,
        "crdld_test_data_accessed": False,
        "frozen_external_accessed": False,
        "real_robot_corridor_established": False,
        "corridor_limitation": "corridor metrics are image proxies derived from merged row labels; robot footprint, camera calibration, traversability, headland and negative-scene ground truth remain unavailable",
    }
    (output_dir / "day63_results_centerline_oof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def finalize_day63_resnet_oof_from_cache(
    *,
    train_root: Path,
    train_manifest: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    train_cache: Path,
    validation_cache: Path,
    comparison_count: int = 8,
) -> dict[str, Any]:
    """Assemble audited results from label-excluding ResNet18 OOF heatmaps."""
    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(
        validation_manifest, "validation_development"
    )
    train_samples = _load_centerline_samples(
        train_root,
        train_stems,
        partition="train_development",
        resolution=192,
    )
    validation_samples = _load_centerline_samples(
        validation_root,
        validation_stems,
        partition="reused_validation_development",
        resolution=192,
    )

    def read_cache(path: Path, expected_stems: list[str]) -> list[np.ndarray]:
        with np.load(path) as payload:
            cached_stems = payload["stems"].astype(str).tolist()
            if cached_stems != expected_stems:
                raise ValueError(f"OOF cache stem order mismatch: {path}")
            probabilities = payload["probabilities"].astype(np.float32)
        if probabilities.shape != (len(expected_stems), 192, 192):
            raise ValueError(f"unexpected OOF probability shape: {probabilities.shape}")
        return list(probabilities)

    train_probabilities = read_cache(train_cache, train_stems)
    validation_probabilities = read_cache(validation_cache, validation_stems)
    decoder = {
        "peak_height": 0.20,
        "peak_prominence": 0.03,
        "peak_distance_norm": 0.06,
    }
    train_summary, train_records = _evaluate_centerline_predictions(
        train_samples,
        train_probabilities,
        [None] * len(train_samples),
        **decoder,
    )
    validation_summary, validation_records = _evaluate_centerline_predictions(
        validation_samples,
        validation_probabilities,
        [None] * len(validation_samples),
        **decoder,
    )
    all_samples = train_samples + validation_samples
    all_probabilities = train_probabilities + validation_probabilities
    overall_summary, all_records = _evaluate_centerline_predictions(
        all_samples,
        all_probabilities,
        [None] * len(all_samples),
        **decoder,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "geometry_metrics_resnet18_oof.csv", all_records)
    ranked = sorted(
        validation_records, key=lambda row: (row["recall"], row["precision"])
    )
    selected = (
        ranked[: comparison_count // 2]
        + ranked[-(comparison_count - comparison_count // 2) :]
    )
    by_key = {sample["key"]: sample for sample in validation_samples}
    contact_rows: list[np.ndarray] = []
    review_stems: list[str] = []
    for row in selected:
        sample = by_key[row["key"]]
        review_stems.append(row["key"])
        overlay = _draw_multirow_prediction(
            sample["image"], sample["centerline_prediction"], sample["reference"]
        )
        mask_bgr = cv2.cvtColor(sample["mask"], cv2.COLOR_GRAY2BGR)
        contact_rows.append(np.hstack((sample["image"], mask_bgr, overlay)))
    if contact_rows:
        cv2.imwrite(
            str(output_dir / "geometry_comparison_resnet18_oof.jpg"),
            np.vstack(contact_rows),
        )
    result = {
        "schema_version": 3,
        "marker": "DAY63_MULTIROW_RELEARNING_COMPLETE",
        "method": "partition_matched_three_fold_resnet18_centerline_oof",
        "input": "RGB plus frozen Day62 morphology mask",
        "decoder": decoder,
        "train_development_oof": train_summary,
        "reused_validation_development_oof": validation_summary,
        "overall_oof": overall_summary,
        "review_stems": review_stems,
        "selection_disclosure": "architecture and decoder were iterated after aggregate access to both development partitions; every reported OOF image label was excluded from its fold model, but these are development results, not untouched confirmation",
        "train_protocol": "three folds; each train-development fold model starts from local ImageNet ResNet18 weights and trains 10 epochs on the other two folds",
        "validation_protocol": "three folds; each validation fold model starts from the train-development model and fine-tunes 8 epochs on the other two validation folds",
        "day61_color_retuned": False,
        "day62_morphology_retuned": False,
        "crdld_test_data_accessed": False,
        "frozen_external_accessed": False,
        "real_robot_corridor_established": False,
        "corridor_limitation": "image-space proxy only; merged masks do not label robot footprint, traversability, headlands, negative scenes or calibrated ground geometry",
        "runtime_limitation": "OOF heatmaps were cached before the final audit; runtime is re-benchmarked separately rather than reconstructed",
    }
    (output_dir / "day63_results_resnet18_oof.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _fit_resnet18_centerline_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    pretrained_state: dict[str, torch.Tensor] | None,
    initial_state: dict[str, torch.Tensor] | None,
    epochs: int,
    seed: int,
    device: str,
    learning_rate: float,
) -> ResNet18RowUNet:
    torch.manual_seed(seed)
    model = ResNet18RowUNet(pretrained_state=pretrained_state).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    target_float = targets.astype(np.float32)
    positives = float(target_float.sum())
    positive_weight = min(
        15.0, float(target_float.size - positives) / max(1.0, positives)
    )
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(features), torch.from_numpy(targets)
        ),
        batch_size=10 if device.startswith("cuda") else 2,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    for _ in range(epochs):
        model.train()
        for inputs, labels in loader:
            inputs = inputs.to(device=device, dtype=torch.float32) / 255.0
            labels = labels.to(device=device, dtype=torch.float32)
            logits = model(inputs)
            probability = torch.sigmoid(logits)
            dice_loss = 1.0 - (
                2.0 * (probability * labels).sum(dim=(1, 2, 3)) + 1.0
            ) / ((probability + labels).sum(dim=(1, 2, 3)) + 1.0)
            loss = bce(logits, labels) + dice_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _generate_resnet18_oof_cache(
    samples: list[dict[str, Any]],
    *,
    cache_path: Path,
    pretrained_state: dict[str, torch.Tensor] | None,
    initial_state: dict[str, torch.Tensor] | None,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> None:
    features = np.asarray([sample["features"] for sample in samples], dtype=np.uint8)
    targets = np.asarray([sample["target"] for sample in samples], dtype=np.uint8)
    stems = [sample["stem"] for sample in samples]
    folds = split_crossfit_folds(stems, fold_count=3)
    stem_to_index = {stem: index for index, stem in enumerate(stems)}
    probabilities = np.zeros((len(samples), 192, 192), dtype=np.float32)
    for fold_index, held_stems in enumerate(folds):
        held_indices = [stem_to_index[stem] for stem in held_stems]
        held_set = set(held_indices)
        fit_indices = [index for index in range(len(samples)) if index not in held_set]
        model = _fit_resnet18_centerline_model(
            features[fit_indices],
            targets[fit_indices],
            pretrained_state=pretrained_state,
            initial_state=initial_state,
            epochs=epochs,
            seed=seed + fold_index,
            device=device,
            learning_rate=learning_rate,
        )
        predicted, _ = _predict_centerline_heatmaps(
            model, features[held_indices], device=device
        )
        probabilities[np.asarray(held_indices)] = predicted
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        probabilities=probabilities,
        stems=np.asarray([sample["stem"] for sample in samples]),
    )


def run_day63_resnet18_oof_study(
    *,
    train_root: Path,
    train_manifest: Path,
    validation_root: Path,
    validation_manifest: Path,
    output_dir: Path,
    pretrained_weights: Path,
    comparison_count: int = 8,
    reuse_existing_cache: bool = True,
) -> dict[str, Any]:
    """Reproduce the partition-matched ResNet18 OOF study and final audit."""
    if not pretrained_weights.is_file():
        raise ValueError(f"missing local ResNet18 weights: {pretrained_weights}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_state = torch.load(
        pretrained_weights, map_location="cpu", weights_only=True
    )
    train_stems = read_manifest_stems(train_manifest, "train_development")
    validation_stems = read_manifest_stems(
        validation_manifest, "validation_development"
    )
    train_samples = _load_centerline_samples(
        train_root,
        train_stems,
        partition="train_development",
        resolution=192,
    )
    validation_samples = _load_centerline_samples(
        validation_root,
        validation_stems,
        partition="reused_validation_development",
        resolution=192,
    )
    train_cache = output_dir / "day63_resnet18_train_oof_probabilities.npz"
    validation_cache = output_dir / "day63_resnet18_validation_oof_probabilities.npz"
    final_model_path = output_dir / "day63_resnet18_centerline_model.pt"
    if not reuse_existing_cache or not train_cache.is_file():
        _generate_resnet18_oof_cache(
            train_samples,
            cache_path=train_cache,
            pretrained_state=pretrained_state,
            initial_state=None,
            epochs=10,
            learning_rate=3e-4,
            seed=6500,
            device=device,
        )
    if not reuse_existing_cache or not final_model_path.is_file():
        features = np.asarray(
            [sample["features"] for sample in train_samples], dtype=np.uint8
        )
        targets = np.asarray(
            [sample["target"] for sample in train_samples], dtype=np.uint8
        )
        train_model = _fit_resnet18_centerline_model(
            features,
            targets,
            pretrained_state=pretrained_state,
            initial_state=None,
            epochs=12,
            seed=63,
            device=device,
            learning_rate=3e-4,
        )
        train_state = {key: value.cpu() for key, value in train_model.state_dict().items()}
        torch.save(
            {
                "state_dict": train_state,
                "resolution": 192,
                "input": "RGB_plus_frozen_Day62_mask",
            },
            final_model_path,
        )
    else:
        train_state = torch.load(
            final_model_path, map_location="cpu", weights_only=True
        )["state_dict"]
    if not reuse_existing_cache or not validation_cache.is_file():
        _generate_resnet18_oof_cache(
            validation_samples,
            cache_path=validation_cache,
            pretrained_state=None,
            initial_state=train_state,
            epochs=8,
            learning_rate=1e-4,
            seed=6400,
            device=device,
        )
    return finalize_day63_resnet_oof_from_cache(
        train_root=train_root,
        train_manifest=train_manifest,
        validation_root=validation_root,
        validation_manifest=validation_manifest,
        output_dir=output_dir,
        train_cache=train_cache,
        validation_cache=validation_cache,
        comparison_count=comparison_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--no-reuse-oof-cache", action="store_true")
    parser.add_argument(
        "--study",
        choices=("v1", "v2", "multirow", "centerline", "resnet_oof"),
        default="v2",
    )
    args = parser.parse_args()
    if args.study == "resnet_oof":
        if args.pretrained_weights is None:
            parser.error("--pretrained-weights is required for --study resnet_oof")
        run_day63_resnet18_oof_study(
            train_root=args.train_root,
            train_manifest=args.train_manifest,
            validation_root=args.validation_root,
            validation_manifest=args.validation_manifest,
            output_dir=args.output_dir,
            pretrained_weights=args.pretrained_weights,
            comparison_count=args.count,
            reuse_existing_cache=not args.no_reuse_oof_cache,
        )
        return 0
    runner = {
        "v1": run_day63_study,
        "v2": run_day63_v2_study,
        "multirow": run_day63_multirow_study,
        "centerline": run_day63_centerline_study,
    }[args.study]
    runner(
        train_root=args.train_root,
        train_manifest=args.train_manifest,
        validation_root=args.validation_root,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        comparison_count=args.count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
