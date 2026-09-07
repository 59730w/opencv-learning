"""Day66 complete offline video pilot for the frozen Day65 crop-row pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch


DAY65_CODE = Path(__file__).resolve().parents[2] / "65_crop_row_video_temporal_stability" / "code"
if str(DAY65_CODE) not in sys.path:
    sys.path.insert(0, str(DAY65_CODE))

from day65_video_temporal import (  # noqa: E402
    FrozenDay63Predictor,
    OpticalFlowObservationBridge,
    TemporalConfig,
    TemporalCorridorTracker,
    filter_plausible_rows,
    load_video_entries,
    prepare_optical_flow_frame,
    prepare_video_feature,
    process_video_episode,
    decode_centerline_heatmap,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OptimizedCpuFrozenDay63Predictor(FrozenDay63Predictor):
    """Numerically equivalent CPU backend optimized without changing model weights."""

    backend = "torchscript_optimize_for_inference_channels_last"

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__(checkpoint_path, device="cpu", batch_size=1)
        self.model = self.model.to(memory_format=torch.channels_last)
        example = torch.zeros(
            (1, 4, self.resolution, self.resolution), dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            traced = torch.jit.trace(self.model, example, check_trace=False)
            self.model = torch.jit.optimize_for_inference(traced.eval())

    def predict(self, frames: list[np.ndarray]) -> list[tuple[Any, ...]]:
        if not frames:
            self.last_runtime_ms_per_frame = 0.0
            return []
        started = time.perf_counter()
        features = np.asarray(
            [prepare_video_feature(frame, resolution=self.resolution) for frame in frames],
            dtype=np.uint8,
        )
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for index in range(len(features)):
                batch = (
                    torch.from_numpy(features[index : index + 1])
                    .to(dtype=torch.float32)
                    .div_(255.0)
                    .contiguous(memory_format=torch.channels_last)
                )
                probabilities.extend(torch.sigmoid(self.model(batch)).numpy()[:, 0])
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
            1000.0 * (time.perf_counter() - started) / len(frames)
        )
        return predictions


def build_execution_predictor(
    checkpoint_path: Path, *, device: str | None, batch_size: int | None
) -> FrozenDay63Predictor:
    """Select the verified optimized backend for CPU and frozen eager CUDA path."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cpu":
        return OptimizedCpuFrozenDay63Predictor(Path(checkpoint_path))
    return FrozenDay63Predictor(
        Path(checkpoint_path), device=resolved_device, batch_size=batch_size
    )


def load_and_verify_frozen_config(
    config_path: Path, day65_result_path: Path
) -> tuple[TemporalConfig, bool, dict[str, Any]]:
    """Load Day66's tracked freeze and prove it matches the accepted Day65 result."""
    config_path = Path(config_path)
    day65_result_path = Path(day65_result_path)
    freeze = json.loads(config_path.read_text(encoding="utf-8"))
    day65 = json.loads(day65_result_path.read_text(encoding="utf-8"))
    selected = freeze.get("selected_config")
    accepted = day65.get("config")
    optical_flow_enabled = freeze.get("optical_flow_enabled")
    if day65.get("marker") != "DAY65_TEMPORAL_DEVELOPMENT_COMPLETE":
        raise ValueError("Day65 result does not carry the accepted completion marker")
    if selected != accepted or optical_flow_enabled != day65.get("optical_flow_enabled"):
        raise ValueError("Day66 frozen configuration does not exactly match Day65 result")
    if day65.get("frozen_video_frames_accessed") is not False:
        raise ValueError("Day65 evidence does not prove frozen-video non-access")
    if freeze.get("allowed_roles") != ["temporal_development", "shifted_development"]:
        raise ValueError("Day66 allowed roles do not match the preregistered development roles")
    config = TemporalConfig(**selected)
    evidence = {
        "exact_day65_match": config.__dict__ == accepted,
        "frozen_config_sha256": _sha256(config_path),
        "day65_result_sha256": _sha256(day65_result_path),
        "allowed_roles": list(freeze["allowed_roles"]),
        "frozen_role": freeze.get("frozen_role"),
    }
    return config, bool(optical_flow_enabled), evidence


def project_navigation_contract(record: dict[str, Any]) -> dict[str, Any]:
    """Project a Day65 diagnostic record into safe Day66 navigation semantics."""
    pilot_state = str(record.get("temporal_status"))
    if pilot_state not in {"valid", "candidate", "degraded", "reject"}:
        raise ValueError(f"unsupported Day65 temporal status: {pilot_state}")
    rows = []
    track_ids = list(record.get("temporal_track_ids") or [])
    temporal_rows = list(record.get("temporal_rows") or [])
    if len(track_ids) != len(temporal_rows):
        raise ValueError("temporal track IDs and rows must be frame-aligned")
    for track_id, row in zip(track_ids, temporal_rows):
        rows.append({"track_id": int(track_id), **dict(row), "status": "tracked"})

    navigation_available = (
        pilot_state == "valid"
        and record.get("temporal_center") is not None
        and record.get("temporal_pair") is not None
    )
    public_status = "valid" if navigation_available else (
        "reject" if pilot_state == "reject" else "degraded"
    )
    left = right = centerline = lateral = heading = vanishing = spacing = None
    if navigation_available:
        pair = [int(item) for item in record["temporal_pair"]]
        indexed = {int(row["track_id"]): row for row in rows}
        if len(pair) != 2 or pair[0] not in indexed or pair[1] not in indexed:
            raise ValueError("valid frame selected corridor IDs are absent from tracked rows")
        left, right = indexed[pair[0]], indexed[pair[1]]
        center_near = (float(left["near_x_norm"]) + float(right["near_x_norm"])) / 2.0
        center_far = (float(left["far_x_norm"]) + float(right["far_x_norm"])) / 2.0
        centerline = [[center_near, 0.90], [center_far, 0.40]]
        lateral = center_near - 0.5
        heading = float(record["temporal_heading"])
        vanishing = record.get("temporal_vanishing_point")
        spacing = float(right["near_x_norm"]) - float(left["near_x_norm"])

    return {
        "episode": record.get("episode"),
        "frame_index": int(record.get("frame_index", -1)),
        "crop_rows": rows,
        "corridor_left_boundary": left,
        "corridor_right_boundary": right,
        "corridor_centerline_points_norm": centerline,
        "lateral_offset_norm": lateral,
        "heading_error_deg": heading,
        "vanishing_point_norm": vanishing,
        "row_spacing_norm": spacing,
        "confidence": float(record.get("temporal_confidence") or 0.0),
        "pilot_state": pilot_state,
        "status": public_status,
        "navigation_available": navigation_available,
        "reason": str(record.get("temporal_reason") or ""),
        "diagnostics": {
            "raw_status": record.get("raw_status"),
            "raw_row_count": int(record.get("raw_row_count", 0)),
            "plausible_row_count": int(record.get("plausible_row_count", 0)),
            "flow_only_row_count": int(record.get("flow_only_row_count", 0)),
            "active_or_selected_track_ids": record.get("temporal_pair"),
        },
    }


def _normalized_point(x_norm: float, y_norm: float, width: int, height: int) -> tuple[int, int]:
    return (
        round(float(x_norm) * (width - 1)),
        round(float(y_norm) * (height - 1)),
    )


def _wrap_overlay_text(
    text: str,
    *,
    max_width: int,
    font_scale: float,
    thickness: int,
    max_lines: int = 2,
) -> list[str]:
    """Wrap short diagnostic prose to a bounded number of OpenCV text lines."""
    words = str(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = words.pop(0)
    while words:
        candidate = f"{current} {words[0]}"
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if width <= max_width or len(lines) >= max_lines - 1:
            current = candidate
            words.pop(0)
        else:
            lines.append(current)
            current = words.pop(0)
    lines.append(current)
    if cv2.getTextSize(
        lines[-1], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )[0][0] > max_width:
        tail = lines[-1]
        while tail and cv2.getTextSize(
            f"{tail}...", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0] > max_width:
            tail = tail[:-1].rstrip()
        lines[-1] = f"{tail}..." if tail else "..."
    return lines


def _put_text_outlined(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw readable HUD text over both dark foliage and bright sky."""
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 0, 0), thickness + max(2, thickness), cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color, thickness, cv2.LINE_AA,
    )


def draw_pilot_overlay(frame: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    """Render diagnostic identities and only contract-valid navigation geometry."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("overlay expects a BGR image with three channels")
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    display_scale = max(1.0, width / 320.0)
    thin = max(1, round(display_scale))
    thick = max(2, round(2.0 * display_scale))
    near_y = 0.90
    far_y = 0.40
    for row in record.get("crop_rows") or []:
        near = _normalized_point(row["near_x_norm"], near_y, width, height)
        far = _normalized_point(row["far_x_norm"], far_y, width, height)
        cv2.line(canvas, near, far, (190, 190, 190), thin, cv2.LINE_AA)
        _put_text_outlined(
            canvas,
            f"ID{int(row['track_id'])}",
            (
                max(0, near[0] - round(10 * display_scale)),
                max(round(12 * display_scale), near[1] - round(4 * display_scale)),
            ),
            font_scale=0.32 * display_scale,
            color=(235, 235, 235),
            thickness=thin,
        )

    if record.get("navigation_available"):
        selected = (
            ("LEFT", record.get("corridor_left_boundary"), (255, 80, 220)),
            ("RIGHT", record.get("corridor_right_boundary"), (255, 180, 40)),
        )
        for boundary_name, row, color in selected:
            if row is None:
                raise ValueError("valid navigation requires both selected boundaries")
            near_point = _normalized_point(
                row["near_x_norm"], near_y, width, height
            )
            far_point = _normalized_point(
                row["far_x_norm"], far_y, width, height
            )
            cv2.line(
                canvas,
                near_point,
                far_point,
                color,
                thick,
                cv2.LINE_AA,
            )
            boundary_label = f"{boundary_name}=ID{int(row['track_id'])}"
            label_scale = 0.30 * display_scale
            label_width = cv2.getTextSize(
                boundary_label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, thin
            )[0][0]
            label_center = (
                (near_point[0] + far_point[0]) // 2,
                (near_point[1] + far_point[1]) // 2,
            )
            _put_text_outlined(
                canvas,
                boundary_label,
                (
                    min(max(0, label_center[0] - label_width // 2), width - label_width - 1),
                    label_center[1],
                ),
                font_scale=label_scale,
                color=color,
                thickness=thin,
            )
        centerline = record.get("corridor_centerline_points_norm")
        if not centerline or len(centerline) != 2:
            raise ValueError("valid navigation requires a two-point centerline")
        cv2.line(
            canvas,
            _normalized_point(*centerline[0], width, height),
            _normalized_point(*centerline[1], width, height),
            (0, 255, 0),
            thick,
            cv2.LINE_AA,
        )
        vanishing = record.get("vanishing_point_norm")
        if vanishing is not None:
            cv2.drawMarker(
                canvas,
                _normalized_point(*vanishing, width, height),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                round(9 * display_scale),
                thin,
                cv2.LINE_AA,
            )
        heading_deg = float(record["heading_error_deg"])
        heading_rad = math.radians(heading_deg)
        direction_origin = (
            width - round(44 * display_scale),
            round(52 * display_scale),
        )
        direction_length = round(22 * display_scale)
        direction_tip = (
            round(direction_origin[0] + math.sin(heading_rad) * direction_length),
            round(direction_origin[1] - math.cos(heading_rad) * direction_length),
        )
        cv2.arrowedLine(
            canvas,
            direction_origin,
            direction_tip,
            (0, 255, 255),
            thick,
            cv2.LINE_AA,
            tipLength=0.28,
        )
        _put_text_outlined(
            canvas,
            "DIR",
            (direction_origin[0] - round(10 * display_scale),
             direction_origin[1] + round(12 * display_scale)),
            font_scale=0.28 * display_scale,
            color=(0, 255, 255),
            thickness=thin,
        )

    pilot_state = str(record.get("pilot_state", "unknown"))
    status_color = {
        "valid": (0, 255, 0),
        "candidate": (0, 210, 255),
        "degraded": (0, 140, 255),
        "reject": (0, 0, 255),
    }.get(pilot_state, (255, 255, 255))
    confidence = float(record.get("confidence") or 0.0)
    top_label = (
        f"F{int(record.get('frame_index', -1)):05d} "
        f"pilot={pilot_state} nav={record.get('status')} conf={confidence:.2f}"
    )
    _put_text_outlined(
        canvas, top_label,
        (round(5 * display_scale), round(13 * display_scale)),
        font_scale=0.34 * display_scale,
        color=status_color,
        thickness=thin,
    )
    diagnostics = record.get("diagnostics") or {}
    heading = record.get("heading_error_deg")
    heading_text = (
        f"{float(heading):+.1f}deg"
        if record.get("navigation_available") and heading is not None
        else "n/a"
    )
    secondary = (
        f"rows={len(record.get('crop_rows') or [])} "
        f"flow={int(diagnostics.get('flow_only_row_count', 0))} "
        f"heading={heading_text}"
    )
    _put_text_outlined(
        canvas, secondary,
        (round(5 * display_scale), round(27 * display_scale)),
        font_scale=0.32 * display_scale,
        color=(230, 230, 230),
        thickness=thin,
    )
    reason = str(record.get("reason") or "")
    if reason:
        reason_scale = 0.30 * display_scale
        reason_lines = _wrap_overlay_text(
            reason,
            max_width=width - round(10 * display_scale),
            font_scale=reason_scale,
            thickness=thin,
        )
        baseline = height - round(6 * display_scale)
        line_step = round(14 * display_scale)
        for index, line in enumerate(reason_lines):
            y = baseline - (len(reason_lines) - 1 - index) * line_step
            _put_text_outlined(
                canvas, line, (round(5 * display_scale), y),
                font_scale=reason_scale,
                color=status_color,
                thickness=thin,
            )
    return canvas


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )


def verify_rendered_video(path: Path, *, expected_frames: int) -> dict[str, Any]:
    """Decode an output video completely instead of trusting container metadata."""
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    decoded = 0
    width = height = 0
    if opened:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded += 1
    capture.release()
    return {
        "opened": opened,
        "decoded_frame_count": decoded,
        "expected_frame_count": int(expected_frames),
        "width": width,
        "height": height,
        "passed": opened and decoded == int(expected_frames) and width > 0 and height > 0,
    }


def _navigation_invariant_violations(records: list[dict[str, Any]]) -> int:
    navigation_fields = (
        "corridor_left_boundary",
        "corridor_right_boundary",
        "corridor_centerline_points_norm",
        "lateral_offset_norm",
        "heading_error_deg",
        "row_spacing_norm",
    )
    violations = 0
    for record in records:
        available = bool(record.get("navigation_available"))
        valid = record.get("pilot_state") == "valid" and record.get("status") == "valid"
        payload_complete = all(record.get(field) is not None for field in navigation_fields)
        if available != valid or (available and not payload_complete):
            violations += 1
        if not available and any(record.get(field) is not None for field in navigation_fields):
            violations += 1
    return violations


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 1 or not items:
        return []
    ordered = sorted(items, key=lambda row: (str(row.get("episode")), int(row.get("frame_index", -1))))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indexes = sorted(
        {round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)}
    )
    return [ordered[index] for index in indexes]


def select_stratified_audit_samples(
    records: list[dict[str, Any]], *, per_group: int = 6
) -> dict[str, list[dict[str, Any]]]:
    """Select deterministic state, transition, and optical-flow review frames."""
    groups = {
        state: [record for record in records if record.get("pilot_state") == state]
        for state in ("valid", "candidate", "degraded", "reject")
    }
    transitions = []
    previous_by_episode: dict[str, str] = {}
    for record in sorted(
        records, key=lambda row: (str(row.get("episode")), int(row.get("frame_index", -1)))
    ):
        episode = str(record.get("episode"))
        state = str(record.get("pilot_state"))
        previous = previous_by_episode.get(episode)
        if previous is not None and previous != state:
            transitions.append(record)
        previous_by_episode[episode] = state
    flow = [
        record
        for record in records
        if int((record.get("diagnostics") or {}).get("flow_only_row_count", 0)) > 0
    ]
    groups["state_transition"] = transitions
    groups["flow_bridge"] = flow
    return {name: _evenly_spaced(items, per_group) for name, items in groups.items()}


def _read_overlay_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open overlay for visual audit: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot decode audit frame {frame_index} from {path}")
    return frame


def build_visual_audit(
    output_dir: Path, *, per_group: int = 6, columns: int = 3
) -> dict[str, Any]:
    """Create deterministic contact sheets from complete Day66 overlay videos."""
    if per_group < 1 or columns < 1:
        raise ValueError("per_group and columns must be positive")
    output_dir = Path(output_dir)
    records = []
    for path in sorted(output_dir.glob("*.jsonl")):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    selected = select_stratified_audit_samples(records, per_group=per_group)
    contact_sheets: dict[str, str] = {}
    selected_metadata: dict[str, list[dict[str, Any]]] = {}
    for group, items in selected.items():
        selected_metadata[group] = [
            {
                "episode": item["episode"],
                "frame_index": int(item["frame_index"]),
                "pilot_state": item["pilot_state"],
                "status": item["status"],
                "reason": item.get("reason"),
            }
            for item in items
        ]
        if not items:
            continue
        frames = [
            _read_overlay_frame(
                output_dir / f"{item['episode']}_day66_overlay.mp4",
                int(item["frame_index"]),
            )
            for item in items
        ]
        height, width = frames[0].shape[:2]
        normalized = [
            frame if frame.shape[:2] == (height, width)
            else cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
        blank = np.zeros_like(normalized[0])
        rows = []
        for start in range(0, len(normalized), columns):
            row = normalized[start : start + columns]
            row.extend([blank.copy() for _ in range(columns - len(row))])
            rows.append(cv2.hconcat(row))
        sheet = cv2.vconcat(rows)
        sheet_path = output_dir / f"day66_audit_{group}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise ValueError(f"cannot write visual audit sheet: {sheet_path}")
        contact_sheets[group] = str(sheet_path)
    result = {
        "schema_version": 1,
        "selection": "deterministic evenly spaced samples in episode/frame order",
        "record_count": len(records),
        "per_group": per_group,
        "selected_counts": {name: len(items) for name, items in selected.items()},
        "selected_frames": selected_metadata,
        "contact_sheets": contact_sheets,
        "evidence_boundary": (
            "Visual review checks interpretability and obvious geometry failures; "
            "it is not framewise accuracy or safety ground truth."
        ),
    }
    (output_dir / "day66_visual_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_episode_pilot(
    entry: dict[str, Any],
    *,
    predictor: Any,
    config: TemporalConfig,
    use_optical_flow: bool,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one complete development episode and emit aligned data and overlay video."""
    if entry.get("role") not in {"temporal_development", "shifted_development"}:
        raise ValueError("Day66 may process development roles only")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(entry["video_path"])
    episode = str(entry["episode"])
    started = time.perf_counter()
    raw_records, day65_summary = process_video_episode(
        video_path,
        predictor=predictor,
        config=config,
        episode=episode,
        use_optical_flow=use_optical_flow,
    )
    perception_seconds = time.perf_counter() - started
    records = []
    for raw in raw_records:
        projected = project_navigation_contract(raw)
        projected["role"] = str(entry["role"])
        records.append(projected)

    jsonl_path = output_dir / f"{episode}.jsonl"
    _write_jsonl(jsonl_path, records)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot reopen source video for overlay rendering: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0:
        fps = 10.0
    overlay_path = output_dir / f"{episode}_day66_overlay.mp4"
    render_scale = 2.0
    render_size = (round(width * render_scale), round(height * render_scale))
    writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, render_size
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"cannot open overlay writer: {overlay_path}")
    render_started = time.perf_counter()
    rendered = 0
    while rendered < len(records):
        ok, frame = capture.read()
        if not ok:
            break
        display_frame = cv2.resize(frame, render_size, interpolation=cv2.INTER_CUBIC)
        writer.write(draw_pilot_overlay(display_frame, records[rendered]))
        rendered += 1
    extra_source_frame = False
    if rendered == len(records):
        extra_source_frame = bool(capture.read()[0])
    capture.release()
    writer.release()
    render_seconds = time.perf_counter() - render_started
    overlay_check = verify_rendered_video(overlay_path, expected_frames=len(records))
    frame_count = len(records)
    summary = dict(day65_summary)
    summary.update(
        {
            "role": str(entry["role"]),
            "source_frame_count": frame_count,
            "rendered_frame_count": rendered,
            "jsonl_record_count": frame_count,
            "source_extra_frame_after_records": extra_source_frame,
            "overlay_path": str(overlay_path),
            "jsonl_path": str(jsonl_path),
            "overlay_sha256": _sha256(overlay_path),
            "jsonl_sha256": _sha256(jsonl_path),
            "source_video_sha256": _sha256(video_path),
            "overlay_decode_complete": bool(overlay_check["passed"]),
            "overlay_verification": overlay_check,
            "perception_seconds": perception_seconds,
            "render_encode_seconds": render_seconds,
            "render_scale": render_scale,
            "render_size": list(render_size),
            "perception_mean_ms_per_frame": 1000.0 * perception_seconds / max(1, frame_count),
            "render_encode_mean_ms_per_frame": 1000.0 * render_seconds / max(1, frame_count),
            "navigation_invariant_violations": _navigation_invariant_violations(records),
        }
    )
    return records, summary


def pilot_acceptance_checks(result: dict[str, Any]) -> dict[str, bool]:
    """Evaluate structural Day66 pilot gates without claiming accuracy."""
    summaries = list(result.get("episode_summaries") or [])
    allowed = {"temporal_development", "shifted_development"}
    return {
        "at_least_one_development_episode": int(result.get("episode_count", 0)) > 0,
        "development_roles_only": set(result.get("roles") or []).issubset(allowed),
        "all_source_videos_decode_complete": bool(summaries) and all(
            bool(item.get("decode_complete")) for item in summaries
        ),
        "all_overlay_videos_decode_complete": bool(summaries) and all(
            bool(item.get("overlay_decode_complete")) for item in summaries
        ),
        "frame_json_overlay_alignment": bool(summaries) and all(
            int(item.get("source_frame_count", -1))
            == int(item.get("rendered_frame_count", -2))
            == int(item.get("jsonl_record_count", -3))
            and not bool(item.get("source_extra_frame_after_records"))
            for item in summaries
        ),
        "frozen_config_exact_day65_match": bool(
            (result.get("frozen_config_evidence") or {}).get("exact_day65_match")
        ),
        "frozen_role_structurally_excluded": bool(
            result.get("frozen_role_structurally_excluded")
        ),
        "frozen_video_frames_not_accessed": result.get("frozen_video_frames_accessed") is False,
        "navigation_outputs_only_when_valid": int(
            result.get("navigation_invariant_violations", -1)
        ) == 0,
    }


def run_day66_pilot(
    *,
    manifest_path: Path,
    frozen_config_path: Path,
    day65_result_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: Any | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the complete development-only offline pilot with frozen Day65 behavior."""
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, optical_flow_enabled, freeze_evidence = load_and_verify_frozen_config(
        frozen_config_path, day65_result_path
    )
    entries = load_video_entries(manifest_path)
    if predictor is None:
        predictor = build_execution_predictor(
            checkpoint_path, device=device, batch_size=batch_size
        )
    summaries = []
    all_records: list[dict[str, Any]] = []
    for entry in entries:
        records, summary = run_episode_pilot(
            entry,
            predictor=predictor,
            config=config,
            use_optical_flow=optical_flow_enabled,
            output_dir=output_dir,
        )
        all_records.extend(records)
        summaries.append(summary)
    status_counts = {
        state: sum(record["pilot_state"] == state for record in all_records)
        for state in ("valid", "candidate", "degraded", "reject")
    }
    public_status_counts = {
        state: sum(record["status"] == state for record in all_records)
        for state in ("valid", "degraded", "reject")
    }
    violations = _navigation_invariant_violations(all_records)
    repo_root = Path(__file__).resolve().parents[2]
    implementation_paths = {
        "day61": repo_root / "61_crop_row_color_illumination" / "code" / "day61_color_illumination.py",
        "day62": repo_root / "62_crop_row_morphology_regions" / "code" / "day62_morphology_regions.py",
        "day63": repo_root / "63_crop_row_geometry_extraction" / "code" / "day63_crop_row_geometry.py",
        "day65": repo_root / "65_crop_row_video_temporal_stability" / "code" / "day65_video_temporal.py",
        "day66": Path(__file__).resolve(),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "marker": "DAY66_OFFLINE_PILOT_INCOMPLETE",
        "method": (
            "frozen Day63 multirow perception plus frozen Day65 optical-flow, "
            "ordered-identity Kalman tracking and reject-aware corridor state machine"
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": (
            _sha256(checkpoint_path) if checkpoint_path.exists() else "INJECTED_TEST_PREDICTOR"
        ),
        "implementation_sha256": {
            name: _sha256(path) for name, path in implementation_paths.items()
        },
        "device": device,
        "config": config.__dict__,
        "optical_flow_enabled": optical_flow_enabled,
        "frozen_config_evidence": freeze_evidence,
        "episode_count": len(entries),
        "roles": sorted({str(entry["role"]) for entry in entries}),
        "aggregate": {
            "frame_count": len(all_records),
            "pilot_state_counts": status_counts,
            "public_status_counts": public_status_counts,
            "valid_fraction": status_counts["valid"] / max(1, len(all_records)),
            "mean_perception_ms_per_frame": (
                1000.0 * sum(item["perception_seconds"] for item in summaries)
                / max(1, len(all_records))
            ),
            "mean_render_encode_ms_per_frame": (
                1000.0 * sum(item["render_encode_seconds"] for item in summaries)
                / max(1, len(all_records))
            ),
        },
        "episode_summaries": summaries,
        "navigation_invariant_violations": violations,
        "frozen_role_structurally_excluded": True,
        "frozen_video_frames_accessed": False,
        "rgb_timestamp_alignment_used": False,
        "crop_lines_used_as_ground_truth": False,
        "real_video_unsafe_false_valid_rate": None,
        "real_video_safety_gate": "BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH",
        "evidence_boundary": (
            "Development videos provide unlabeled pilot evidence only. Output completeness "
            "and state-machine invariants do not establish real-video navigation accuracy."
        ),
    }
    checks = pilot_acceptance_checks(result)
    result["acceptance_checks"] = checks
    if all(checks.values()):
        result["marker"] = "DAY66_OFFLINE_PILOT_COMPLETE"
    (output_dir / "day66_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _read_canonical_frames(video_path: Path, max_frames: int) -> list[np.ndarray]:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open CPU benchmark video: {video_path}")
    frames = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR))
    capture.release()
    if not frames:
        raise ValueError("CPU benchmark video decoded zero frames")
    return frames


def _benchmark_frame(
    frame: np.ndarray,
    *,
    predictor: Any,
    tracker: TemporalCorridorTracker,
    bridge: OpticalFlowObservationBridge | None,
    config: TemporalConfig,
) -> None:
    rows = predictor.predict([frame])
    if len(rows) != 1:
        raise ValueError("benchmark predictor must return one observation sequence")
    plausible = filter_plausible_rows(
        rows[0], minimum_confidence=config.minimum_row_confidence
    )
    if bridge is not None:
        observations, _ = bridge.update(prepare_optical_flow_frame(frame), plausible)
    else:
        observations = plausible
    tracker.update(observations)


def benchmark_canonical_cpu(
    *,
    video_path: Path,
    checkpoint_path: Path,
    config: TemporalConfig,
    use_optical_flow: bool,
    max_frames: int = 120,
    warmup_frames: int = 10,
    cpu_threads: int = 4,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Benchmark per-frame 640x360 perception with declared exclusions."""
    if warmup_frames < 0 or cpu_threads < 1:
        raise ValueError("warmup_frames must be non-negative and cpu_threads positive")
    frames = _read_canonical_frames(Path(video_path), max_frames)
    previous_torch_threads = torch.get_num_threads()
    previous_cv_threads = cv2.getNumThreads()
    torch.set_num_threads(cpu_threads)
    cv2.setNumThreads(cpu_threads)
    try:
        if predictor is None:
            predictor = OptimizedCpuFrozenDay63Predictor(Path(checkpoint_path))
        warm_tracker = TemporalCorridorTracker(config)
        warm_bridge = OpticalFlowObservationBridge(max_flow_age=2) if use_optical_flow else None
        with torch.inference_mode():
            for index in range(warmup_frames):
                _benchmark_frame(
                    frames[index % len(frames)], predictor=predictor,
                    tracker=warm_tracker, bridge=warm_bridge, config=config
                )
            tracker = TemporalCorridorTracker(config)
            bridge = OpticalFlowObservationBridge(max_flow_age=2) if use_optical_flow else None
            samples_ms = []
            for frame in frames:
                started = time.perf_counter()
                _benchmark_frame(
                    frame, predictor=predictor, tracker=tracker,
                    bridge=bridge, config=config
                )
                samples_ms.append(1000.0 * (time.perf_counter() - started))
    finally:
        torch.set_num_threads(previous_torch_threads)
        cv2.setNumThreads(previous_cv_threads)
    median_ms = float(np.median(samples_ms))
    p95_ms = float(np.percentile(samples_ms, 95))
    return {
        "device": "cpu",
        "canonical_size": [640, 360],
        "cpu_threads": cpu_threads,
        "warmup_frames": warmup_frames,
        "measured_frames": len(frames),
        "timing_sample_count": len(samples_ms),
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "minimum_ms": float(min(samples_ms)),
        "maximum_ms": float(max(samples_ms)),
        "target_median_ms": 50.0,
        "target_median_at_most_50ms_passed": median_ms <= 50.0,
        "timing_boundary": (
            "single-frame frozen-model inference plus plausibility filtering, "
            "optical-flow bridge and temporal tracking"
        ),
        "decode_resize_inside_timing": False,
        "overlay_encode_inside_timing": False,
        "optical_flow_max_working_size": [320, 180],
        "batch_size": 1,
        "execution_backend": getattr(predictor, "backend", "injected_predictor"),
        "source_video": str(video_path),
        "source_video_sha256": _sha256(Path(video_path)),
        "evidence_scope": "machine-specific runtime evidence; not robot real-time proof",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a Day66 CLI that deliberately exposes no Day65 tuning controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--day65-result", type=Path, required=True)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=Path(__file__).with_name("day66_frozen_config.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu-benchmark-video", type=Path)
    parser.add_argument("--cpu-benchmark-frames", type=int, default=120)
    parser.add_argument("--cpu-benchmark-warmup", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_day66_pilot(
        manifest_path=args.manifest,
        frozen_config_path=args.frozen_config,
        day65_result_path=args.day65_result,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    if args.cpu_benchmark_video is not None:
        config, optical_flow_enabled, _ = load_and_verify_frozen_config(
            args.frozen_config, args.day65_result
        )
        runtime = benchmark_canonical_cpu(
            video_path=args.cpu_benchmark_video,
            checkpoint_path=args.checkpoint,
            config=config,
            use_optical_flow=optical_flow_enabled,
            max_frames=args.cpu_benchmark_frames,
            warmup_frames=args.cpu_benchmark_warmup,
            cpu_threads=args.cpu_threads,
        )
        result["cpu_runtime_benchmark"] = runtime
        result["runtime_gate_passed"] = runtime["target_median_at_most_50ms_passed"]
        result["day66_engineering_gate_passed"] = (
            all(result["acceptance_checks"].values()) and result["runtime_gate_passed"]
        )
        (Path(args.output_dir) / "day66_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        result["runtime_gate_passed"] = None
        result["day66_engineering_gate_passed"] = None
    print(
        json.dumps(
            {
                "marker": result["marker"],
                "acceptance_checks": result["acceptance_checks"],
                "runtime_gate_passed": result.get("runtime_gate_passed"),
                "real_video_safety_gate": result["real_video_safety_gate"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(result["acceptance_checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
