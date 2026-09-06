"""Day65 multi-row temporal tracking and reject-aware corridor state machine.

The temporal layer consumes frozen Day63 crop-row observations and preserves
the Day64 image-coordinate contract.  It never upgrades a frame with a missing
left or right observed boundary to ``valid``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import cv2
import torch


DAY63_CODE = Path(__file__).resolve().parents[2] / "63_crop_row_geometry_extraction" / "code"
DAY64_CODE = Path(__file__).resolve().parents[2] / "64_crop_row_camera_coordinates_measurement" / "code"
for dependency in (DAY63_CODE, DAY64_CODE):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from day63_crop_row_geometry import (  # noqa: E402
    CropRowLine,
    ResNet18RowUNet,
    _frozen_day62_mask,
    decode_centerline_heatmap,
    prepare_centerline_tensor,
)
from day64_camera_measurement import (  # noqa: E402
    CORRIDOR_AUDIT_Y_NORM,
    multirow_coordinate_measurement,
)


DEVELOPMENT_VIDEO_ROLES = frozenset(
    {"temporal_development", "shifted_development"}
)


@dataclass(frozen=True)
class TemporalConfig:
    association_gate: float = 0.13
    max_missing_frames: int = 4
    confirm_frames: int = 3
    recovery_frames: int = 2
    switch_confirm_frames: int = 4
    confirmation_window_frames: int = 4
    process_variance: float = 2.5e-5
    measurement_variance: float = 1.5e-3
    minimum_row_confidence: float = 0.12

    def __post_init__(self) -> None:
        if self.association_gate <= 0:
            raise ValueError("association_gate must be positive")
        if min(
            self.max_missing_frames,
            self.confirm_frames,
            self.recovery_frames,
            self.switch_confirm_frames,
            self.confirmation_window_frames,
        ) < 1:
            raise ValueError("frame-count thresholds must be positive")


@dataclass(frozen=True)
class TemporalFrameResult:
    status: str
    navigation_available: bool
    track_ids: tuple[int, ...]
    corridor_track_ids: tuple[int, int] | None
    rows: tuple[CropRowLine, ...]
    corridor_center_near_x_norm: float | None
    corridor_center_far_x_norm: float | None
    heading_proxy_deg: float | None
    vanishing_point_norm: tuple[float, float] | None
    confidence: float
    reason: str


@dataclass
class _KalmanRowTrack:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    confidence: float
    support_band_count: int
    hits: int = 1
    missing_frames: int = 0

    @classmethod
    def create(cls, track_id: int, row: CropRowLine) -> "_KalmanRowTrack":
        return cls(
            track_id=track_id,
            state=np.asarray(
                [row.near_x_norm, row.far_x_norm, 0.0, 0.0], dtype=np.float64
            ),
            covariance=np.diag([0.004, 0.004, 0.02, 0.02]).astype(np.float64),
            confidence=float(row.confidence),
            support_band_count=int(row.support_band_count),
        )

    def predict(self, config: TemporalConfig) -> None:
        transition = np.asarray(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
             [0.0, 0.0, 0.72, 0.0], [0.0, 0.0, 0.0, 0.72]],
            dtype=np.float64,
        )
        process = np.diag(
            [config.process_variance, config.process_variance,
             4.0 * config.process_variance, 4.0 * config.process_variance]
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

    def update(self, row: CropRowLine, config: TemporalConfig) -> None:
        observation = np.asarray([row.near_x_norm, row.far_x_norm], dtype=np.float64)
        model = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        variance = config.measurement_variance / max(0.10, float(row.confidence))
        noise = np.eye(2, dtype=np.float64) * variance
        innovation = observation - model @ self.state
        innovation_covariance = model @ self.covariance @ model.T + noise
        gain = self.covariance @ model.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - gain @ model) @ self.covariance
        self.confidence = 0.65 * float(row.confidence) + 0.35 * self.confidence
        self.support_band_count = int(row.support_band_count)
        self.hits += 1
        self.missing_frames = 0

    def mark_missing(self) -> None:
        self.missing_frames += 1
        self.confidence *= 0.72

    def as_row(self) -> CropRowLine:
        return CropRowLine(
            near_x_norm=float(self.state[0]),
            far_x_norm=float(self.state[1]),
            confidence=float(np.clip(self.confidence, 0.0, 1.0)),
            support_band_count=self.support_band_count,
        )


def load_video_entries(manifest_path: Path) -> list[dict[str, Any]]:
    """Load development entries while structurally excluding the frozen role."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("video manifest must contain an episodes list")
    return [dict(item) for item in episodes if item.get("role") in DEVELOPMENT_VIDEO_ROLES]


def filter_plausible_rows(
    rows: Sequence[CropRowLine], *, minimum_confidence: float = 0.12
) -> tuple[CropRowLine, ...]:
    """Reject rows whose evaluated endpoints cannot describe an in-frame row."""
    kept = []
    for row in rows:
        values = (row.near_x_norm, row.far_x_norm, row.confidence)
        if not all(math.isfinite(float(value)) for value in values):
            continue
        if not (-0.05 <= row.near_x_norm <= 1.05):
            continue
        if not (-0.05 <= row.far_x_norm <= 1.05):
            continue
        if not (-0.03 <= row.x_at(CORRIDOR_AUDIT_Y_NORM) <= 1.03):
            continue
        if row.confidence < minimum_confidence:
            continue
        if abs(row.heading_deg) > 40.0:
            continue
        kept.append(row)
    return tuple(sorted(kept, key=lambda item: item.x_at(CORRIDOR_AUDIT_Y_NORM)))


def prepare_video_feature(image: np.ndarray, *, resolution: int = 192) -> np.ndarray:
    """Reuse the frozen Day63 RGB plus Day62-mask inference contract."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image")
    mask = _frozen_day62_mask(image)
    dummy_label = np.zeros(image.shape[:2], dtype=np.uint8)
    feature, _ = prepare_centerline_tensor(
        image, mask, dummy_label, resolution=resolution
    )
    return feature


def propagate_rows_optical_flow(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    rows: Sequence[CropRowLine],
    *,
    forward_backward_limit_px: float = 1.5,
    minimum_points: int = 6,
) -> tuple[CropRowLine, ...]:
    """Propagate line observations using local pyramidal LK feature tracks."""
    if previous_frame.shape != current_frame.shape or previous_frame.ndim != 3:
        raise ValueError("optical-flow frames must have identical BGR shapes")
    height, width = previous_frame.shape[:2]
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    propagated: list[CropRowLine] = []
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    for row in rows:
        mask = np.zeros((height, width), dtype=np.uint8)
        near = (
            round(row.near_x_norm * (width - 1)),
            round(0.90 * (height - 1)),
        )
        far = (
            round(row.far_x_norm * (width - 1)),
            round(0.40 * (height - 1)),
        )
        cv2.line(mask, near, far, 255, max(9, round(0.045 * width)))
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=4,
            mask=mask,
            blockSize=5,
        )
        if points is None or len(points) < minimum_points:
            continue
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None, **lk
        )
        if forward is None or forward_status is None:
            continue
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, forward, None, **lk
        )
        if backward is None or backward_status is None:
            continue
        original = points.reshape(-1, 2)
        moved = forward.reshape(-1, 2)
        returned = backward.reshape(-1, 2)
        good = (
            (forward_status.reshape(-1) > 0)
            & (backward_status.reshape(-1) > 0)
            & (np.linalg.norm(returned - original, axis=1) <= forward_backward_limit_px)
            & (moved[:, 0] >= 0)
            & (moved[:, 0] <= width - 1)
            & (moved[:, 1] >= 0)
            & (moved[:, 1] <= height - 1)
        )
        moved = moved[good]
        original = original[good]
        if len(moved) < minimum_points:
            continue
        y_norm = moved[:, 1] / max(1, height - 1)
        x_norm = moved[:, 0] / max(1, width - 1)
        slope, intercept = np.polyfit(y_norm, x_norm, 1)
        residuals = np.abs(x_norm - (slope * y_norm + intercept))
        keep = residuals <= max(0.025, 2.5 * float(np.median(residuals)))
        if np.count_nonzero(keep) < minimum_points:
            continue
        slope, intercept = np.polyfit(y_norm[keep], x_norm[keep], 1)
        near_x = float(slope * 0.90 + intercept)
        far_x = float(slope * 0.40 + intercept)
        median_fb = float(
            np.median(np.linalg.norm(returned[good][keep] - original[keep], axis=1))
        )
        point_quality = min(1.0, np.count_nonzero(keep) / 15.0)
        confidence = (
            row.confidence
            * 0.90
            * point_quality
            * math.exp(-median_fb / max(forward_backward_limit_px, 1e-6))
        )
        candidate = CropRowLine(
            near_x_norm=near_x,
            far_x_norm=far_x,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            support_band_count=int(np.count_nonzero(keep)),
        )
        propagated.extend(filter_plausible_rows((candidate,), minimum_confidence=0.08))
    return tuple(sorted(propagated, key=lambda item: item.x_at(CORRIDOR_AUDIT_Y_NORM)))


class OpticalFlowObservationBridge:
    """Fuse fresh semantic detections with at most a few LK-propagated frames."""

    def __init__(self, *, max_flow_age: int = 2, match_gate: float = 0.12) -> None:
        if max_flow_age < 1 or match_gate <= 0:
            raise ValueError("optical-flow bridge limits must be positive")
        self.max_flow_age = max_flow_age
        self.match_gate = match_gate
        self._previous_frame: np.ndarray | None = None
        self._states: list[tuple[CropRowLine, int]] = []

    def update(
        self, frame: np.ndarray, detected_rows: Sequence[CropRowLine]
    ) -> tuple[tuple[CropRowLine, ...], int]:
        detected = list(filter_plausible_rows(detected_rows))
        propagated: list[tuple[CropRowLine, int]] = []
        if self._previous_frame is not None:
            for previous_row, age in self._states:
                next_age = age + 1
                if next_age > self.max_flow_age:
                    continue
                moved = propagate_rows_optical_flow(
                    self._previous_frame, frame, (previous_row,)
                )
                if moved:
                    propagated.append((moved[0], next_age))

        states: list[tuple[CropRowLine, int]] = []
        used_propagated: set[int] = set()
        for detection in detected:
            candidates = []
            for index, (flow_row, _) in enumerate(propagated):
                if index in used_propagated:
                    continue
                cost = 0.6 * abs(detection.near_x_norm - flow_row.near_x_norm)
                cost += 0.4 * abs(detection.far_x_norm - flow_row.far_x_norm)
                candidates.append((cost, index, flow_row))
            if candidates and min(candidates)[0] <= self.match_gate:
                _, index, flow_row = min(candidates)
                used_propagated.add(index)
                fused = CropRowLine(
                    near_x_norm=0.75 * detection.near_x_norm + 0.25 * flow_row.near_x_norm,
                    far_x_norm=0.75 * detection.far_x_norm + 0.25 * flow_row.far_x_norm,
                    confidence=max(detection.confidence, 0.8 * flow_row.confidence),
                    support_band_count=max(
                        detection.support_band_count, flow_row.support_band_count
                    ),
                )
                states.append((fused, 0))
            else:
                states.append((detection, 0))
        states.extend(
            item for index, item in enumerate(propagated) if index not in used_propagated
        )

        deduplicated: list[tuple[CropRowLine, int]] = []
        for candidate in sorted(
            states, key=lambda item: item[0].x_at(CORRIDOR_AUDIT_Y_NORM)
        ):
            if (
                deduplicated
                and abs(
                    candidate[0].x_at(CORRIDOR_AUDIT_Y_NORM)
                    - deduplicated[-1][0].x_at(CORRIDOR_AUDIT_Y_NORM)
                ) < 0.035
            ):
                if candidate[0].confidence > deduplicated[-1][0].confidence:
                    deduplicated[-1] = candidate
            else:
                deduplicated.append(candidate)
        self._previous_frame = frame.copy()
        self._states = deduplicated
        rows = tuple(item[0] for item in deduplicated)
        flow_only_count = sum(age > 0 for _, age in deduplicated)
        return rows, flow_only_count


class FrozenDay63Predictor:
    """Batch inference adapter for the accepted Day63 ResNet18 checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise ValueError(f"missing frozen Day63 checkpoint: {checkpoint_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("input") != "RGB_plus_frozen_Day62_mask":
            raise ValueError("checkpoint does not declare the frozen four-channel input")
        self.resolution = int(payload.get("resolution", 192))
        self.model = ResNet18RowUNet()
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.batch_size = batch_size or (32 if self.device.startswith("cuda") else 2)
        self.last_runtime_ms_per_frame: float | None = None

    def predict(self, frames: list[np.ndarray]) -> list[tuple[CropRowLine, ...]]:
        if not frames:
            self.last_runtime_ms_per_frame = 0.0
            return []
        start = time.perf_counter()
        features = np.asarray(
            [prepare_video_feature(frame, resolution=self.resolution) for frame in frames],
            dtype=np.uint8,
        )
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for index in range(0, len(features), self.batch_size):
                batch = torch.from_numpy(features[index : index + self.batch_size]).to(
                    device=self.device, dtype=torch.float32
                ) / 255.0
                probabilities.extend(
                    torch.sigmoid(self.model(batch)).cpu().numpy()[:, 0]
                )
        predictions = [
            decode_centerline_heatmap(
                probability,
                peak_height=0.20,
                peak_prominence=0.03,
                peak_distance_norm=0.06,
            ).rows
            for probability in probabilities
        ]
        self.last_runtime_ms_per_frame = (
            (time.perf_counter() - start) * 1000.0 / len(frames)
        )
        return predictions


def _association_cost(track: _KalmanRowTrack, row: CropRowLine) -> float:
    predicted = track.as_row()
    heading_cost = abs(predicted.heading_deg - row.heading_deg) / 45.0
    return float(
        0.55 * abs(predicted.near_x_norm - row.near_x_norm)
        + 0.35 * abs(predicted.far_x_norm - row.far_x_norm)
        + 0.10 * heading_cost
    )


def _ordered_matches(
    tracks: Sequence[_KalmanRowTrack],
    rows: Sequence[CropRowLine],
    *,
    gate: float,
) -> list[tuple[int, int]]:
    """Sequence alignment that cannot reverse left-to-right row identities."""
    count_tracks, count_rows = len(tracks), len(rows)
    skip_cost = gate * 0.70
    costs = np.full((count_tracks + 1, count_rows + 1), np.inf, dtype=np.float64)
    action = np.empty((count_tracks + 1, count_rows + 1), dtype=object)
    costs[0, 0] = 0.0
    for i in range(count_tracks + 1):
        for j in range(count_rows + 1):
            current = costs[i, j]
            if not math.isfinite(float(current)):
                continue
            if i < count_tracks and current + skip_cost < costs[i + 1, j]:
                costs[i + 1, j] = current + skip_cost
                action[i + 1, j] = (i, j, "skip_track")
            if j < count_rows and current + skip_cost < costs[i, j + 1]:
                costs[i, j + 1] = current + skip_cost
                action[i, j + 1] = (i, j, "new_row")
            if i < count_tracks and j < count_rows:
                match_cost = _association_cost(tracks[i], rows[j])
                if match_cost <= gate and current + match_cost <= costs[i + 1, j + 1]:
                    costs[i + 1, j + 1] = current + match_cost
                    action[i + 1, j + 1] = (i, j, "match")
    matches: list[tuple[int, int]] = []
    i, j = count_tracks, count_rows
    while i or j:
        previous = action[i, j]
        if previous is None:
            break
        old_i, old_j, kind = previous
        if kind == "match":
            matches.append((old_i, old_j))
        i, j = old_i, old_j
    matches.reverse()
    offsets = {row_index - track_index for track_index, row_index in matches}
    if (
        count_tracks == count_rows
        and matches
        and len(offsets) == 1
        and next(iter(offsets)) != 0
    ):
        # A whole-lattice jump of one rank is observationally ambiguous: the
        # nearest rows could be adjacent physical rows.  Starting candidates
        # is safer than silently transferring every identity by one row.
        return []
    return matches


def temporal_jitter(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(array))))


def _same_episode(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("episode", "__single__") == right.get("episode", "__single__")


def _status_switch_count(records: Sequence[dict[str, Any]], key: str) -> int:
    return sum(
        _same_episode(left, right) and left[key] != right[key]
        for left, right in zip(records, records[1:])
    )


def _pair_switch_count(records: Sequence[dict[str, Any]], key: str) -> int:
    count = 0
    previous: tuple[int, ...] | None = None
    previous_episode: str | None = None
    for record in records:
        episode = str(record.get("episode", "__single__"))
        if previous_episode is not None and episode != previous_episode:
            previous = None
        previous_episode = episode
        pair = record.get(key)
        if pair is None:
            continue
        current = tuple(map(int, pair))
        if previous is not None and current != previous:
            count += 1
        previous = current
    return count


def _consecutive_jitter(records: Sequence[dict[str, Any]], key: str) -> float:
    differences = [
        abs(float(right[key]) - float(left[key]))
        for left, right in zip(records, records[1:])
        if _same_episode(left, right)
        and left.get(key) is not None
        and right.get(key) is not None
    ]
    return float(np.median(differences)) if differences else 0.0


def summarize_temporal_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observable video stability without inventing ground truth."""
    count = len(records)
    raw_statuses = [str(row["raw_status"]) for row in records]
    temporal_statuses = [str(row["temporal_status"]) for row in records]
    return {
        "frame_count": count,
        "raw_valid_fraction": raw_statuses.count("valid") / max(1, count),
        "temporal_valid_fraction": temporal_statuses.count("valid") / max(1, count),
        "raw_status_switches": _status_switch_count(records, "raw_status"),
        "temporal_status_switches": _status_switch_count(records, "temporal_status"),
        "raw_corridor_switches": _pair_switch_count(records, "raw_pair"),
        "temporal_corridor_switches": _pair_switch_count(records, "temporal_pair"),
        "raw_center_jitter_norm": _consecutive_jitter(records, "raw_center"),
        "temporal_center_jitter_norm": _consecutive_jitter(records, "temporal_center"),
    }


def run_synthetic_acceptance_study(config: TemporalConfig) -> dict[str, Any]:
    """Exercise declared safety transitions on deterministic labelled scenarios."""
    rng = np.random.default_rng(6501)
    tracker = TemporalCorridorTracker(config)
    raw_centers: list[float] = []
    filtered_centers: list[float] = []
    for noise in rng.normal(0.0, 0.020, 120):
        observations = (
            CropRowLine(0.38 + noise, 0.18 + noise, 0.9, 8),
            CropRowLine(0.46 + noise, 0.38 + noise, 0.9, 8),
            CropRowLine(0.54 + noise, 0.62 + noise, 0.9, 8),
            CropRowLine(0.62 + noise, 0.82 + noise, 0.9, 8),
        )
        raw_centers.append((observations[1].near_x_norm + observations[2].near_x_norm) / 2.0)
        result = tracker.update(observations)
        if result.navigation_available:
            filtered_centers.append(float(result.corridor_center_near_x_norm))

    unsupported: list[tuple[str, tuple[CropRowLine, ...]]] = []
    left_only = (
        CropRowLine(0.38, 0.18, 0.9, 8),
        CropRowLine(0.46, 0.38, 0.9, 8),
    )
    central = (
        CropRowLine(0.40, 0.20, 0.9, 8),
        CropRowLine(0.50, 0.50, 0.9, 8),
        CropRowLine(0.60, 0.80, 0.9, 8),
    )
    unsupported.extend(("one_side_missing", left_only) for _ in range(10))
    unsupported.extend(("central_row", central) for _ in range(10))
    unsupported.extend(("no_rows", ()) for _ in range(8))
    statuses: list[tuple[str, str]] = []
    for kind, observations in unsupported:
        statuses.append((kind, tracker.update(observations).status))

    raw_jitter = temporal_jitter(raw_centers)
    filtered_jitter = temporal_jitter(filtered_centers)
    return {
        "stable_frame_count": len(raw_centers),
        "unsupported_frame_count": len(unsupported),
        "raw_center_jitter_norm": raw_jitter,
        "temporal_center_jitter_norm": filtered_jitter,
        "jitter_reduction_fraction": 1.0 - filtered_jitter / max(raw_jitter, 1e-12),
        "unsafe_false_valid_rate": (
            sum(status == "valid" for _, status in statuses) / len(statuses)
        ),
        "one_side_missing_valid_count": sum(
            kind == "one_side_missing" and status == "valid"
            for kind, status in statuses
        ),
        "central_row_valid_count": sum(
            kind == "central_row" and status == "valid" for kind, status in statuses
        ),
        "status_counts": {
            status: sum(item_status == status for _, item_status in statuses)
            for status in ("valid", "candidate", "degraded", "reject")
        },
    }


def temporal_acceptance_checks(
    aggregate: dict[str, Any], synthetic: dict[str, Any]
) -> dict[str, bool]:
    """Apply predeclared Day65 engineering gates without claiming video GT."""
    raw_valid = float(aggregate["raw_valid_fraction"])
    raw_jitter = float(aggregate["raw_center_jitter_norm"])
    raw_status_switches = int(aggregate["raw_status_switches"])
    raw_corridor_switches = int(aggregate["raw_corridor_switches"])
    return {
        "retains_at_least_30pct_of_raw_valid_frames": (
            float(aggregate["temporal_valid_fraction"]) / max(raw_valid, 1e-12) >= 0.30
        ),
        "center_jitter_reduction_at_least_50pct": (
            1.0 - float(aggregate["temporal_center_jitter_norm"]) / max(raw_jitter, 1e-12)
            >= 0.50
        ),
        "status_switch_reduction_at_least_50pct": (
            1.0 - int(aggregate["temporal_status_switches"]) / max(raw_status_switches, 1)
            >= 0.50
        ),
        "corridor_switch_reduction_at_least_30pct": (
            1.0 - int(aggregate["temporal_corridor_switches"]) / max(raw_corridor_switches, 1)
            >= 0.30
        ),
        "synthetic_unsafe_false_valid_at_most_0_05": (
            float(synthetic["unsafe_false_valid_rate"]) <= 0.05
        ),
        "one_side_missing_never_valid": int(synthetic["one_side_missing_valid_count"]) == 0,
        "central_crop_row_never_valid": int(synthetic["central_row_valid_count"]) == 0,
    }


class TemporalCorridorTracker:
    """Track every plausible row and gate corridor validity with hysteresis."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()
        self._tracks: list[_KalmanRowTrack] = []
        self._next_track_id = 0
        self._active_pair: tuple[int, int] | None = None
        self._pending_pair: tuple[int, int] | None = None
        self._pending_count = 0
        self._pending_age = 0
        self._interrupted = False
        self._recovery_count = 0
        self._last_corridor_signature: tuple[float, float, float, float] | None = None
        self._reidentify_pair: tuple[int, int] | None = None
        self._reidentify_count = 0

    def update(self, rows: Sequence[CropRowLine]) -> TemporalFrameResult:
        observations = filter_plausible_rows(
            rows, minimum_confidence=self.config.minimum_row_confidence
        )
        self._tracks.sort(key=lambda track: track.as_row().x_at(CORRIDOR_AUDIT_Y_NORM))
        for track in self._tracks:
            track.predict(self.config)
        matches = _ordered_matches(
            self._tracks, observations, gate=self.config.association_gate
        )
        matched_tracks = {track_index for track_index, _ in matches}
        matched_rows = {row_index for _, row_index in matches}
        for track_index, row_index in matches:
            self._tracks[track_index].update(observations[row_index], self.config)
        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.mark_missing()
        for index, observation in enumerate(observations):
            if index not in matched_rows:
                self._tracks.append(
                    _KalmanRowTrack.create(self._next_track_id, observation)
                )
                self._next_track_id += 1
        self._tracks = [
            track
            for track in self._tracks
            if track.missing_frames <= self.config.max_missing_frames
        ]
        observed_tracks = sorted(
            (track for track in self._tracks if track.missing_frames == 0),
            key=lambda track: track.as_row().x_at(CORRIDOR_AUDIT_Y_NORM),
        )
        smoothed_rows = tuple(track.as_row() for track in observed_tracks)
        track_ids = tuple(track.track_id for track in observed_tracks)
        measurement = multirow_coordinate_measurement(
            rows=smoothed_rows, width=640, height=360
        )
        candidate_pair = None
        candidate_signature = None
        if (
            measurement.status == "valid"
            and measurement.corridor_left_index is not None
            and measurement.corridor_right_index is not None
        ):
            candidate_pair = (
                observed_tracks[measurement.corridor_left_index].track_id,
                observed_tracks[measurement.corridor_right_index].track_id,
            )
            candidate_signature = (
                float(measurement.corridor_center_near_x_norm),
                float(measurement.corridor_center_far_x_norm),
                float(smoothed_rows[measurement.corridor_left_index].near_x_norm),
                float(smoothed_rows[measurement.corridor_right_index].near_x_norm),
            )

        status, reason = self._transition(
            candidate_pair, measurement.reason, candidate_signature
        )
        navigation_available = status == "valid" and candidate_pair is not None
        if navigation_available:
            self._last_corridor_signature = candidate_signature
            center_near = measurement.corridor_center_near_x_norm
            center_far = measurement.corridor_center_far_x_norm
            heading = measurement.heading_proxy_deg
            vanishing = measurement.vanishing_point_norm
            confidence = measurement.confidence
        else:
            center_near = center_far = heading = None
            vanishing = None
            confidence = float(np.mean([row.confidence for row in smoothed_rows])) if smoothed_rows else 0.0
        return TemporalFrameResult(
            status=status,
            navigation_available=navigation_available,
            track_ids=track_ids,
            corridor_track_ids=candidate_pair if navigation_available else self._active_pair,
            rows=smoothed_rows,
            corridor_center_near_x_norm=center_near,
            corridor_center_far_x_norm=center_far,
            heading_proxy_deg=heading,
            vanishing_point_norm=vanishing,
            confidence=float(confidence),
            reason=reason,
        )

    def _transition(
        self,
        candidate_pair: tuple[int, int] | None,
        measurement_reason: str,
        candidate_signature: tuple[float, float, float, float] | None,
    ) -> tuple[str, str]:
        if candidate_pair is None:
            self._reidentify_pair = None
            self._reidentify_count = 0
            if self._pending_pair is not None:
                self._pending_age += 1
                if self._pending_age >= self.config.confirmation_window_frames:
                    self._pending_pair = None
                    self._pending_count = 0
                    self._pending_age = 0
            self._recovery_count = 0
            if self._active_pair is not None:
                self._interrupted = True
            if not self._tracks:
                self._active_pair = None
                return "reject", "no live crop-row tracks remain"
            return "degraded", measurement_reason

        if self._active_pair is None:
            if self._same_corridor_signature(candidate_signature):
                if candidate_pair == self._reidentify_pair:
                    self._reidentify_count += 1
                else:
                    self._reidentify_pair = candidate_pair
                    self._reidentify_count = 1
                if self._reidentify_count >= self.config.recovery_frames:
                    self._active_pair = candidate_pair
                    self._reidentify_pair = None
                    self._reidentify_count = 0
                    return "valid", "same corridor geometry reidentified with new row tracks"
                return "candidate", "same corridor geometry awaits reidentification confirmation"
            if candidate_pair == self._pending_pair:
                self._pending_count += 1
            else:
                self._pending_pair = candidate_pair
                self._pending_count = 1
            self._pending_age = 0
            if self._pending_count >= self.config.confirm_frames:
                self._active_pair = candidate_pair
                self._pending_pair = None
                self._pending_count = 0
                self._pending_age = 0
                return "valid", "initial corridor confirmed across consecutive frames"
            return "candidate", "corridor candidate awaits consecutive confirmation"

        if candidate_pair == self._active_pair:
            self._pending_pair = None
            self._pending_count = 0
            self._pending_age = 0
            if self._interrupted:
                self._recovery_count += 1
                if self._recovery_count < self.config.recovery_frames:
                    return "candidate", "known corridor is recovering after interruption"
                self._interrupted = False
                self._recovery_count = 0
            return "valid", "same observed boundary identities remain confirmed"

        self._interrupted = True
        self._recovery_count = 0
        if candidate_pair == self._pending_pair:
            self._pending_count += 1
        else:
            self._pending_pair = candidate_pair
            self._pending_count = 1
        self._pending_age = 0
        if self._pending_count >= self.config.switch_confirm_frames:
            self._active_pair = candidate_pair
            self._pending_pair = None
            self._pending_count = 0
            self._pending_age = 0
            self._interrupted = False
            return "valid", "new corridor confirmed before identity switch"
        return "degraded", "different corridor awaits switch confirmation"

    def _same_corridor_signature(
        self, candidate: tuple[float, float, float, float] | None
    ) -> bool:
        previous = self._last_corridor_signature
        if previous is None or candidate is None:
            return False
        differences = np.abs(np.asarray(previous) - np.asarray(candidate))
        return bool(
            differences[0] <= 0.08
            and differences[1] <= 0.08
            and differences[2] <= 0.12
            and differences[3] <= 0.12
        )


def process_video_episode(
    video_path: Path,
    *,
    predictor: Any,
    config: TemporalConfig,
    episode: str,
    use_optical_flow: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode one complete episode and apply raw and temporal measurements."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    observations = predictor.predict(frames)
    if len(observations) != len(frames):
        raise ValueError("predictor must return one row sequence per decoded frame")

    tracker = TemporalCorridorTracker(config)
    bridge = OpticalFlowObservationBridge(max_flow_age=2) if use_optical_flow else None
    records: list[dict[str, Any]] = []
    for frame_index, raw_rows in enumerate(observations):
        plausible = filter_plausible_rows(
            raw_rows, minimum_confidence=config.minimum_row_confidence
        )
        raw_measurement = multirow_coordinate_measurement(
            rows=plausible, width=frames[frame_index].shape[1],
            height=frames[frame_index].shape[0]
        )
        raw_pair = None
        if (
            raw_measurement.status == "valid"
            and raw_measurement.corridor_left_index is not None
            and raw_measurement.corridor_right_index is not None
        ):
            raw_pair = [
                raw_measurement.corridor_left_index,
                raw_measurement.corridor_right_index,
            ]
        if bridge is None:
            temporal_observations = plausible
            flow_only_count = 0
        else:
            temporal_observations, flow_only_count = bridge.update(
                frames[frame_index], plausible
            )
        temporal = tracker.update(temporal_observations)
        records.append(
            {
                "episode": episode,
                "frame_index": frame_index,
                "raw_row_count": len(raw_rows),
                "plausible_row_count": len(plausible),
                "filtered_row_count": len(raw_rows) - len(plausible),
                "flow_only_row_count": flow_only_count,
                "plausible_rows": [
                    {
                        "near_x_norm": row.near_x_norm,
                        "far_x_norm": row.far_x_norm,
                        "confidence": row.confidence,
                        "support_band_count": row.support_band_count,
                    }
                    for row in plausible
                ],
                "temporal_observation_rows": [
                    {
                        "near_x_norm": row.near_x_norm,
                        "far_x_norm": row.far_x_norm,
                        "confidence": row.confidence,
                        "support_band_count": row.support_band_count,
                    }
                    for row in temporal_observations
                ],
                "raw_status": raw_measurement.status,
                "raw_center": raw_measurement.corridor_center_near_x_norm,
                "raw_heading": raw_measurement.heading_proxy_deg,
                "raw_vanishing_point": raw_measurement.vanishing_point_norm,
                "raw_pair": raw_pair,
                "temporal_status": temporal.status,
                "temporal_center": temporal.corridor_center_near_x_norm,
                "temporal_heading": temporal.heading_proxy_deg,
                "temporal_vanishing_point": temporal.vanishing_point_norm,
                "temporal_pair": list(temporal.corridor_track_ids)
                if temporal.corridor_track_ids is not None
                else None,
                "temporal_track_ids": list(temporal.track_ids),
                "temporal_rows": [
                    {
                        "near_x_norm": row.near_x_norm,
                        "far_x_norm": row.far_x_norm,
                        "confidence": row.confidence,
                        "support_band_count": row.support_band_count,
                    }
                    for row in temporal.rows
                ],
                "temporal_confidence": temporal.confidence,
                "temporal_reason": temporal.reason,
            }
        )
    summary = summarize_temporal_records(records)
    summary.update(
        {
            "episode": episode,
            "video_path": str(video_path),
            "reported_frame_count": expected_frames,
            "decode_complete": len(frames) == expected_frames,
            "optical_flow_enabled": use_optical_flow,
            "filtered_row_fraction": (
                sum(record["filtered_row_count"] for record in records)
                / max(1, sum(record["raw_row_count"] for record in records))
            ),
        }
    )
    return records, summary


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )


def draw_temporal_overlay(image: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    """Draw raw rows, filtered tracks, and only a valid navigation center."""
    canvas = image.copy()
    height, width = canvas.shape[:2]

    def draw_rows(items: Sequence[dict[str, Any]], color: tuple[int, int, int], thickness: int) -> None:
        for item in items:
            near = (round(float(item["near_x_norm"]) * (width - 1)), round(0.90 * (height - 1)))
            far = (round(float(item["far_x_norm"]) * (width - 1)), round(0.40 * (height - 1)))
            cv2.line(canvas, near, far, color, thickness, cv2.LINE_AA)

    draw_rows(record.get("plausible_rows", []), (0, 255, 255), 1)
    draw_rows(record.get("temporal_rows", []), (255, 255, 0), 2)
    if record.get("temporal_status") == "valid" and record.get("temporal_center") is not None:
        near_x = float(record["temporal_center"])
        heading = math.radians(float(record.get("temporal_heading") or 0.0))
        far_x = near_x + math.tan(heading) * 0.50
        cv2.line(
            canvas,
            (round(near_x * (width - 1)), round(0.90 * (height - 1))),
            (round(far_x * (width - 1)), round(0.40 * (height - 1))),
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
    status = str(record.get("temporal_status", "unknown"))
    status_color = {
        "valid": (0, 255, 0),
        "candidate": (0, 200, 255),
        "degraded": (0, 128, 255),
        "reject": (0, 0, 255),
    }.get(status, (255, 255, 255))
    label = (
        f"raw={record.get('raw_status')} temporal={status} "
        f"flow_rows={record.get('flow_only_row_count', 0)}"
    )
    cv2.putText(
        canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        status_color, 1, cv2.LINE_AA
    )
    return canvas


def run_day65_video_study(
    *,
    manifest_path: Path,
    output_dir: Path,
    predictor: Any,
    config: TemporalConfig,
    use_optical_flow: bool = False,
) -> dict[str, Any]:
    """Run only predeclared development roles and preserve the frozen boundary."""
    entries = load_video_entries(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for entry in entries:
        records, summary = process_video_episode(
            Path(entry["video_path"]),
            predictor=predictor,
            config=config,
            episode=str(entry["episode"]),
            use_optical_flow=use_optical_flow,
        )
        for record in records:
            record["role"] = entry["role"]
        _write_jsonl(output_dir / f"{entry['episode']}.jsonl", records)
        episode_summaries.append(summary | {"role": entry["role"]})
        all_records.extend(records)

    aggregate = summarize_temporal_records(all_records)
    synthetic = run_synthetic_acceptance_study(config)
    engineering_checks = temporal_acceptance_checks(aggregate, synthetic)
    result = {
        "schema_version": 1,
        "marker": "DAY65_TEMPORAL_DEVELOPMENT_COMPLETE",
        "method": "ordered multi-row identity tracking plus confidence-weighted Kalman filtering and reject-aware corridor hysteresis",
        "config": config.__dict__,
        "episode_count": len(entries),
        "roles": sorted({entry["role"] for entry in entries}),
        "aggregate": aggregate,
        "episode_summaries": episode_summaries,
        "synthetic_acceptance": synthetic,
        "engineering_acceptance_checks": engineering_checks,
        "engineering_gate_passed": all(engineering_checks.values()),
        "all_videos_decode_complete": all(
            summary["decode_complete"] for summary in episode_summaries
        ),
        "frozen_video_frames_accessed": False,
        "frozen_role_structurally_excluded": True,
        "rgb_timestamp_alignment_used": False,
        "optical_flow_enabled": use_optical_flow,
        "crop_lines_used_as_ground_truth": False,
        "real_video_unsafe_false_valid_rate": None,
        "real_video_safety_gate": "BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH",
        "evidence_boundary": (
            "CROW development videos provide unlabeled temporal and OOD-development evidence; "
            "synthetic scenarios verify state-machine invariants but do not establish real-video safety accuracy."
        ),
    }
    (output_dir / "day65_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the reproducible Day65 development-study command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--association-gate", type=float, default=0.25)
    parser.add_argument("--max-missing-frames", type=int, default=8)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--recovery-frames", type=int, default=1)
    parser.add_argument("--switch-confirm-frames", type=int, default=6)
    parser.add_argument("--confirmation-window-frames", type=int, default=8)
    parser.add_argument("--process-variance", type=float, default=1e-5)
    parser.add_argument("--measurement-variance", type=float, default=0.0025)
    parser.add_argument("--minimum-row-confidence", type=float, default=0.12)
    parser.add_argument(
        "--no-optical-flow",
        action="store_false",
        dest="use_optical_flow",
        help="disable the accepted two-frame optical-flow observation bridge",
    )
    parser.set_defaults(use_optical_flow=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = TemporalConfig(
        association_gate=args.association_gate,
        max_missing_frames=args.max_missing_frames,
        confirm_frames=args.confirm_frames,
        recovery_frames=args.recovery_frames,
        switch_confirm_frames=args.switch_confirm_frames,
        confirmation_window_frames=args.confirmation_window_frames,
        process_variance=args.process_variance,
        measurement_variance=args.measurement_variance,
        minimum_row_confidence=args.minimum_row_confidence,
    )
    predictor = FrozenDay63Predictor(
        args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    result = run_day65_video_study(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        predictor=predictor,
        config=config,
        use_optical_flow=args.use_optical_flow,
    )
    print(json.dumps({
        "marker": result["marker"],
        "engineering_gate_passed": result["engineering_gate_passed"],
        "real_video_safety_gate": result["real_video_safety_gate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
