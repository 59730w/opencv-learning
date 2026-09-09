"""Day68 severe-occlusion recovery guard for the frozen crop-row pilot.

The controlled change is a causal one-frame interlock.  After a prolonged
non-valid interval, a newly valid corridor with weak boundary support is kept
diagnostic-only only when the proposed corridor is also poorly observable or a
boundary leaves the image. Tracking rows and stable IDs remain available, but
public navigation geometry is cleared whenever the guard fires.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


DAY66_CODE = Path(__file__).resolve().parents[2] / "66_crop_row_offline_video_pilot" / "code"
if str(DAY66_CODE) not in sys.path:
    sys.path.insert(0, str(DAY66_CODE))
from day66_offline_video_pilot import draw_pilot_overlay, verify_rendered_video  # noqa: E402


ALLOWED_DEVELOPMENT_ROLES = {"temporal_development", "shifted_development"}
NAVIGATION_FIELDS = (
    "corridor_left_boundary",
    "corridor_right_boundary",
    "corridor_centerline_points_norm",
    "lateral_offset_norm",
    "heading_error_deg",
    "vanishing_point_norm",
    "row_spacing_norm",
)
EVENT_SPAN_PATTERN = re.compile(r"__(\d{6})_(\d{6})__")


@dataclass(frozen=True)
class OcclusionGuardConfig:
    """Frozen, interpretable Day68 thresholds."""

    selected_cause: str = "severe_occlusion"
    min_preceding_nonvalid_frames: int = 8
    max_min_boundary_support: int = 6
    maximum_corridor_edge_fraction: float = 0.22
    minimum_valid_retention: float = 0.90
    evidence_status: str = "MODEL_ASSISTED_REVIEW_DEVELOPMENT_ONLY"

    def __post_init__(self) -> None:
        if self.selected_cause != "severe_occlusion":
            raise ValueError("Day68 may change only the preregistered severe_occlusion cause")
        if self.min_preceding_nonvalid_frames < 1:
            raise ValueError("min_preceding_nonvalid_frames must be positive")
        if self.max_min_boundary_support < 1:
            raise ValueError("max_min_boundary_support must be positive")
        if not 0.0 <= self.maximum_corridor_edge_fraction <= 1.0:
            raise ValueError("maximum_corridor_edge_fraction must be in [0, 1]")
        if not 0.0 < self.minimum_valid_retention <= 1.0:
            raise ValueError("minimum_valid_retention must be in (0, 1]")


def _navigation_violation(record: dict[str, Any]) -> bool:
    navigation = bool(record.get("navigation_available"))
    state = str(record.get("pilot_state"))
    fields_present = any(record.get(field) is not None for field in NAVIGATION_FIELDS)
    return (state != "valid" and (navigation or fields_present)) or (
        state == "valid" and (not navigation or not fields_present)
    )


def _minimum_boundary_support(record: dict[str, Any]) -> int:
    left = record.get("corridor_left_boundary")
    right = record.get("corridor_right_boundary")
    if left is None or right is None:
        raise ValueError("valid Day66 input requires both corridor boundaries")
    return min(int(left["support_band_count"]), int(right["support_band_count"]))


def _corridor_observability(
    frame: np.ndarray, record: dict[str, Any]
) -> tuple[float, bool]:
    """Return causal image-edge support inside the proposed Day66 corridor."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Day68 observability requires a non-empty BGR frame")
    left = record.get("corridor_left_boundary")
    right = record.get("corridor_right_boundary")
    if left is None or right is None:
        raise ValueError("corridor observability requires both boundaries")
    coords = [
        float(left["near_x_norm"]),
        float(left["far_x_norm"]),
        float(right["near_x_norm"]),
        float(right["far_x_norm"]),
    ]
    boundary_out_of_frame = any(value < 0.0 or value > 1.0 for value in coords)
    height, width = frame.shape[:2]
    points = np.asarray(
        [
            [coords[1] * (width - 1), 0.40 * (height - 1)],
            [coords[3] * (width - 1), 0.40 * (height - 1)],
            [coords[2] * (width - 1), 0.90 * (height - 1)],
            [coords[0] * (width - 1), 0.90 * (height - 1)],
        ],
        dtype=np.float32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points.astype(np.int32), 255)
    area = mask > 0
    if not np.any(area):
        return 0.0, boundary_out_of_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140) > 0
    return float(edges[area].mean()), boundary_out_of_frame


class OcclusionRecoveryGuard:
    """Apply the causal Day68 guard independently to one ordered episode."""

    def __init__(self, config: OcclusionGuardConfig) -> None:
        self.config = config
        self._episode: str | None = None
        self._preceding_nonvalid_frames = 0

    def apply(
        self, record: dict[str, Any], frame: np.ndarray | None = None
    ) -> dict[str, Any]:
        result = copy.deepcopy(record)
        episode = str(result.get("episode"))
        if self._episode is None or episode != self._episode:
            self._episode = episode
            self._preceding_nonvalid_frames = 0

        if _navigation_violation(result):
            raise ValueError("Day66 input violates valid-only navigation semantics")

        baseline_state = str(result["pilot_state"])
        prior_nonvalid = self._preceding_nonvalid_frames
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics.update(
            {
                "day68_baseline_pilot_state": baseline_state,
                "day68_baseline_reason": str(result.get("reason") or ""),
                "preceding_nonvalid_frames": int(prior_nonvalid),
                "minimum_boundary_support": None,
                "corridor_edge_fraction": None,
                "boundary_out_of_frame": None,
                "maximum_corridor_edge_fraction": self.config.maximum_corridor_edge_fraction,
                "day68_guard_triggered": False,
                "occlusion_risk": 0.0,
            }
        )

        if baseline_state != "valid":
            self._preceding_nonvalid_frames += 1
            result["diagnostics"] = diagnostics
            return result

        minimum_support = _minimum_boundary_support(result)
        prolonged_outage = prior_nonvalid >= self.config.min_preceding_nonvalid_frames
        weak_reentry = minimum_support <= self.config.max_min_boundary_support
        edge_fraction: float | None = None
        boundary_out_of_frame: bool | None = None
        if prolonged_outage and weak_reentry:
            if frame is None:
                raise ValueError("a BGR frame is required for a weak recovery candidate")
            edge_fraction, boundary_out_of_frame = _corridor_observability(frame, result)
        poorly_observable = (
            edge_fraction is not None
            and edge_fraction <= self.config.maximum_corridor_edge_fraction
        )
        triggered = prolonged_outage and weak_reentry and bool(
            boundary_out_of_frame or poorly_observable
        )
        diagnostics["minimum_boundary_support"] = minimum_support
        diagnostics["corridor_edge_fraction"] = edge_fraction
        diagnostics["boundary_out_of_frame"] = boundary_out_of_frame
        diagnostics["day68_guard_triggered"] = triggered
        diagnostics["occlusion_risk"] = 1.0 if triggered else 0.0

        # This is deliberately a one-frame interlock. The next Day66-valid frame
        # is judged from a fresh recovery state rather than being rejected forever.
        self._preceding_nonvalid_frames = 0
        if triggered:
            result["pilot_state"] = "degraded"
            result["status"] = "degraded"
            result["navigation_available"] = False
            for field in NAVIGATION_FIELDS:
                result[field] = None
            result["reason"] = (
                "severe_occlusion_recovery_guard: weak boundary support after "
                f"{prior_nonvalid} non-valid frames; corridor edge fraction="
                f"{edge_fraction:.4f}, boundary_out_of_frame={boundary_out_of_frame}"
            )
        result["diagnostics"] = diagnostics
        return result


def apply_guard_to_records(
    records: Sequence[dict[str, Any]],
    config: OcclusionGuardConfig,
    *,
    frames: Sequence[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    if frames is not None and len(records) != len(frames):
        raise ValueError("records and frames must be frame-aligned")
    guard = OcclusionRecoveryGuard(config)
    return [
        guard.apply(record, None if frames is None else frames[index])
        for index, record in enumerate(records)
    ]


def assert_manifest_excludes_frozen(manifest: dict[str, Any]) -> int:
    entries = list(manifest.get("episodes") or [])
    forbidden = [row for row in entries if row.get("role") not in ALLOWED_DEVELOPMENT_ROLES]
    if forbidden:
        raise ValueError("Day68 inputs must exclude every frozen or unknown manifest role")
    return len(entries)


def _event_span(event_id: str) -> tuple[int, int]:
    match = EVENT_SPAN_PATTERN.search(event_id)
    if match is None:
        raise ValueError(f"cannot parse event frame span: {event_id}")
    return int(match.group(1)), int(match.group(2))


def summarize_day68(
    baseline_records: Sequence[dict[str, Any]],
    guarded_records: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    *,
    minimum_valid_retention: float,
) -> dict[str, Any]:
    if len(baseline_records) != len(guarded_records):
        raise ValueError("baseline and guarded records must be frame-aligned")
    baseline_index = {
        (str(row["episode"]), int(row["frame_index"])): row for row in baseline_records
    }
    guarded_index = {
        (str(row["episode"]), int(row["frame_index"])): row for row in guarded_records
    }
    if baseline_index.keys() != guarded_index.keys():
        raise ValueError("baseline and guarded frame keys differ")

    targets = [
        row
        for row in annotations
        if row.get("primary_visual_label") == "severe_occlusion"
        and row.get("navigation_assessment") == "suspected_unsafe"
    ]

    def target_valid_count(index: dict[tuple[str, int], dict[str, Any]]) -> int:
        count = 0
        for target in targets:
            start, end = _event_span(str(target["event_id"]))
            if any(
                bool(index[(str(target["episode"]), frame)]["navigation_available"])
                for frame in range(start, end + 1)
            ):
                count += 1
        return count

    baseline_valid = sum(bool(row["navigation_available"]) for row in baseline_records)
    guarded_valid = sum(bool(row["navigation_available"]) for row in guarded_records)
    retention = guarded_valid / max(1, baseline_valid)
    baseline_unsafe = target_valid_count(baseline_index)
    guarded_unsafe = target_valid_count(guarded_index)
    violations = sum(_navigation_violation(row) for row in guarded_records)
    triggered = [
        row for row in guarded_records if row.get("diagnostics", {}).get("day68_guard_triggered")
    ]
    reviewed_spans: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    for annotation in annotations:
        start, end = _event_span(str(annotation["event_id"]))
        reviewed_spans.setdefault(str(annotation["episode"]), []).append(
            (start, end, annotation)
        )
    by_visual_label: dict[str, int] = {}
    by_navigation_assessment: dict[str, int] = {}
    unmatched = 0
    for row in triggered:
        frame = int(row["frame_index"])
        matches = [
            annotation
            for start, end, annotation in reviewed_spans.get(str(row["episode"]), [])
            if start <= frame <= end
        ]
        if len(matches) != 1:
            unmatched += 1
            continue
        annotation = matches[0]
        visual = str(annotation.get("primary_visual_label"))
        navigation = str(annotation.get("navigation_assessment"))
        by_visual_label[visual] = by_visual_label.get(visual, 0) + 1
        by_navigation_assessment[navigation] = (
            by_navigation_assessment.get(navigation, 0) + 1
        )
    checks = {
        "target_unsafe_valid_reduced": guarded_unsafe < baseline_unsafe,
        "all_reviewed_severe_unsafe_valid_blocked": guarded_unsafe == 0,
        "valid_frame_retention_at_least_floor": retention >= minimum_valid_retention,
        "nonvalid_navigation_leakage_is_zero": violations == 0,
    }
    return {
        "baseline_frame_count": len(baseline_records),
        "baseline_valid_frame_count": baseline_valid,
        "guarded_valid_frame_count": guarded_valid,
        "valid_frame_retention": retention,
        "guard_triggered_frame_count": len(triggered),
        "guard_triggered_episode_count": len({row["episode"] for row in triggered}),
        "guard_triggered_review_breakdown": {
            "by_visual_label": dict(sorted(by_visual_label.items())),
            "by_navigation_assessment": dict(sorted(by_navigation_assessment.items())),
            "unmatched_frame_count": unmatched,
        },
        "baseline_suspected_unsafe_severe_valid_event_count": baseline_unsafe,
        "guarded_suspected_unsafe_severe_valid_event_count": guarded_unsafe,
        "navigation_invariant_violations": violations,
        "guarded_status_counts": {
            state: sum(row["pilot_state"] == state for row in guarded_records)
            for state in ("valid", "candidate", "degraded", "reject")
        },
        "acceptance_checks": checks,
        "day68_engineering_gate_passed": all(checks.values()),
    }


def draw_day68_overlay(frame: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    canvas = draw_pilot_overlay(frame, record)
    _, width = canvas.shape[:2]
    diagnostics = record.get("diagnostics") or {}
    risk = float(diagnostics.get("occlusion_risk") or 0.0)
    triggered = bool(diagnostics.get("day68_guard_triggered"))
    edge = diagnostics.get("corridor_edge_fraction")
    edge_text = "n/a" if edge is None else f"{float(edge):.3f}"
    out = diagnostics.get("boundary_out_of_frame")
    out_text = "n/a" if out is None else ("yes" if out else "no")
    support = diagnostics.get("minimum_boundary_support")
    support_text = "n/a" if support is None else str(int(support))
    text = (
        f"D68 risk={risk:.2f} guard={'ON' if triggered else 'OFF'} "
        f"prior={int(diagnostics.get('preceding_nonvalid_frames') or 0)} "
        f"support={support_text} edge={edge_text} out={out_text}"
    )
    scale = max(0.28, width / 1140.0)
    thickness = max(1, round(width / 640.0))
    origin = (5, max(39, round(42 * max(1.0, width / 320.0))))
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 255) if triggered else (220, 220, 220),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> OcclusionGuardConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = {key: payload[key] for key in asdict(OcclusionGuardConfig())}
    return OcclusionGuardConfig(**fields)


def _verify_declared_hashes(
    payload: dict[str, Any], paths: dict[str, Path]
) -> dict[str, str]:
    declared = dict(payload.get("frozen_inputs") or {})
    actual = {name: _sha256(path) for name, path in paths.items()}
    for name, expected in declared.items():
        if name not in actual:
            raise ValueError(f"unknown frozen input hash key: {name}")
        if str(expected) != actual[name]:
            raise ValueError(f"frozen input hash mismatch: {name}")
    return actual


def _process_episode(
    video_path: Path,
    records: Sequence[dict[str, Any]],
    config: OcclusionGuardConfig,
    overlay_path: Path | None,
) -> tuple[list[dict[str, Any]], int]:
    """Synchronously decode, guard, and optionally render one development video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open development video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 10.0
    writer: cv2.VideoWriter | None = None
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"cannot create overlay video: {overlay_path}")
    guard = OcclusionRecoveryGuard(config)
    guarded: list[dict[str, Any]] = []
    count = 0
    try:
        for expected, record in enumerate(records):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"video ended before JSONL at frame {expected}: {video_path}")
            if int(record["frame_index"]) != expected:
                raise ValueError("Day66 JSONL frame indices must be zero-based and contiguous")
            guarded_record = guard.apply(record, frame)
            guarded.append(guarded_record)
            if writer is not None:
                writer.write(draw_day68_overlay(frame, guarded_record))
            count += 1
        ok, _ = capture.read()
        if ok:
            raise ValueError(f"video has more frames than JSONL: {video_path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return guarded, count


def run_day68(
    *,
    day66_output: Path,
    manifest_path: Path,
    annotations_path: Path,
    day67_results_path: Path,
    config_path: Path,
    output_dir: Path,
    render_overlays: bool,
) -> dict[str, Any]:
    """Apply the frozen guard to every declared development episode."""
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    day67_results = json.loads(day67_results_path.read_text(encoding="utf-8"))
    if day67_results.get("marker") != "DAY67_FAILURE_TAXONOMY_COMPLETE":
        raise ValueError("Day68 requires the completed Day67 v3 result")
    packet = day67_results.get("packet_metadata") or {}
    if int(packet.get("frozen_frames_accessed", day67_results.get("frozen_frames_accessed", -1))) != 0:
        raise ValueError("Day67 evidence indicates frozen frames were accessed")

    selected_entries = [
        row for row in manifest.get("episodes", [])
        if row.get("role") in ALLOWED_DEVELOPMENT_ROLES
    ]
    assert_manifest_excludes_frozen({"episodes": selected_entries})
    selected = {str(row["episode"]): row for row in selected_entries}
    jsonl_paths = {
        path.stem: path for path in day66_output.glob("*.jsonl")
    }
    if set(jsonl_paths) != set(selected):
        missing = sorted(set(selected) - set(jsonl_paths))
        extra = sorted(set(jsonl_paths) - set(selected))
        raise ValueError(f"Day66 development episode mismatch; missing={missing}, extra={extra}")

    input_hashes = _verify_declared_hashes(
        config_payload,
        {
            "manifest_sha256": manifest_path,
            "day67_annotations_sha256": annotations_path,
            "day67_results_sha256": day67_results_path,
        },
    )
    upstream_day66_expected = config_payload.get("upstream_day66_result_sha256")
    if upstream_day66_expected is not None:
        upstream_day66_path = day66_output / "day66_results.json"
        if not upstream_day66_path.is_file():
            raise ValueError("declared upstream Day66 result is missing")
        upstream_day66_actual = _sha256(upstream_day66_path)
        if str(upstream_day66_expected) != upstream_day66_actual:
            raise ValueError("upstream Day66 result hash mismatch")
        input_hashes["upstream_day66_result_sha256"] = upstream_day66_actual
    output_dir.mkdir(parents=True, exist_ok=True)
    all_baseline: list[dict[str, Any]] = []
    all_guarded: list[dict[str, Any]] = []
    rendered_frame_count = 0
    episode_results = []
    for episode in sorted(selected):
        baseline = _read_jsonl(jsonl_paths[episode])
        if any(str(row.get("episode")) != episode for row in baseline):
            raise ValueError(f"JSONL episode field mismatch: {episode}")
        overlay_path = (
            output_dir / f"{episode}_day68_overlay.mp4" if render_overlays else None
        )
        guarded, decoded = _process_episode(
            Path(selected[episode]["video_path"]), baseline, config, overlay_path
        )
        output_jsonl = output_dir / f"{episode}_day68.jsonl"
        _write_jsonl(output_jsonl, guarded)
        episode_result = {
            "episode": episode,
            "role": selected[episode]["role"],
            "frame_count": len(guarded),
            "guard_triggered_frame_count": sum(
                bool(row["diagnostics"]["day68_guard_triggered"]) for row in guarded
            ),
            "output_jsonl": str(output_jsonl),
            "output_jsonl_sha256": _sha256(output_jsonl),
        }
        if render_overlays:
            assert overlay_path is not None
            rendered_frame_count += decoded
            decode_check = verify_rendered_video(
                overlay_path, expected_frames=len(guarded)
            )
            episode_result.update(
                {
                    "overlay_video": str(overlay_path),
                    "overlay_video_sha256": _sha256(overlay_path),
                    "overlay_frame_count": decoded,
                    "overlay_decode_check": decode_check,
                }
            )
        episode_results.append(episode_result)
        all_baseline.extend(baseline)
        all_guarded.extend(guarded)

    annotations = _read_jsonl(annotations_path)
    summary = summarize_day68(
        all_baseline,
        all_guarded,
        annotations,
        minimum_valid_retention=config.minimum_valid_retention,
    )
    checks = dict(summary["acceptance_checks"])
    checks.update(
        {
            "development_episode_inventory_exact": len(episode_results) == len(selected),
            "all_frames_written": sum(row["frame_count"] for row in episode_results) == len(all_guarded),
            "overlay_frames_aligned": (not render_overlays) or rendered_frame_count == len(all_guarded),
            "all_overlay_videos_decode_exactly": (not render_overlays)
            or all(
                bool(row["overlay_decode_check"]["passed"])
                for row in episode_results
            ),
            "frozen_video_frames_accessed_is_zero": True,
        }
    )
    result = {
        "schema_version": 1,
        "marker": "DAY68_SEVERE_OCCLUSION_COMPLETE",
        "method": "causal_recovery_plus_corridor_observability_interlock_v2",
        "selected_cause": config.selected_cause,
        "evidence_status": config.evidence_status,
        "config": asdict(config),
        "config_sha256": _sha256(config_path),
        "input_hashes": input_hashes,
        "development_episode_count": len(episode_results),
        "development_frame_count": len(all_guarded),
        "frozen_video_frames_accessed": 0,
        "episode_results": episode_results,
        **{key: value for key, value in summary.items() if key not in {"acceptance_checks", "day68_engineering_gate_passed"}},
        "acceptance_checks": checks,
        "day68_engineering_gate_passed": all(checks.values()),
        "real_video_safety_gate": "BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH",
        "claim_boundary": (
            "Day68 uses frozen model-assisted development review, not independent human "
            "framewise corridor-validity ground truth or external safety evidence."
        ),
    }
    result_path = output_dir / "day68_results.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day66-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--day67-annotations", type=Path, required=True)
    parser.add_argument("--day67-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-overlays", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_day68(
        day66_output=args.day66_output,
        manifest_path=args.manifest,
        annotations_path=args.day67_annotations,
        day67_results_path=args.day67_results,
        config_path=args.config,
        output_dir=args.output_dir,
        render_overlays=args.render_overlays,
    )
    print(json.dumps({"marker": result["marker"], "passed": result["day68_engineering_gate_passed"]}))
    return 0 if result["day68_engineering_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
