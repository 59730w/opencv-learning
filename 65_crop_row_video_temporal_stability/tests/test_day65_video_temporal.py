from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import cv2
import pytest


DAY63_CODE = Path(__file__).resolve().parents[2] / "63_crop_row_geometry_extraction" / "code"
DAY65_CODE = Path(__file__).resolve().parents[1] / "code"
for dependency in (DAY63_CODE, DAY65_CODE):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from day63_crop_row_geometry import CropRowLine  # noqa: E402
from day65_video_temporal import (  # noqa: E402
    build_arg_parser,
    FrozenDay63Predictor,
    OpticalFlowObservationBridge,
    TemporalConfig,
    TemporalCorridorTracker,
    filter_plausible_rows,
    load_video_entries,
    process_video_episode,
    propagate_rows_optical_flow,
    prepare_video_feature,
    prepare_optical_flow_frame,
    run_day65_video_study,
    temporal_acceptance_checks,
    draw_temporal_overlay,
    run_synthetic_acceptance_study,
    summarize_temporal_records,
    temporal_jitter,
)


def test_cli_defaults_reproduce_selected_day65_configuration(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--manifest", str(tmp_path / "manifest.json"),
            "--checkpoint", str(tmp_path / "model.pt"),
            "--output-dir", str(tmp_path / "results"),
        ]
    )

    assert args.association_gate == 0.25
    assert args.confirm_frames == 2
    assert args.switch_confirm_frames == 6
    assert args.use_optical_flow is True


def row(near: float, far: float, confidence: float = 0.9) -> CropRowLine:
    return CropRowLine(
        near_x_norm=near,
        far_x_norm=far,
        confidence=confidence,
        support_band_count=8,
    )


def corridor_rows(shift: float = 0.0) -> tuple[CropRowLine, ...]:
    return (
        row(0.18 + shift, 0.38 + shift),
        row(0.38 + shift, 0.46 + shift),
        row(0.62 + shift, 0.54 + shift),
        row(0.82 + shift, 0.62 + shift),
    )


def test_video_loader_excludes_frozen_holdout_by_default(tmp_path: Path) -> None:
    manifest = {
        "episodes": [
            {"episode": "dev", "role": "temporal_development", "video_path": "a.mp4"},
            {"episode": "shift", "role": "shifted_development", "video_path": "b.mp4"},
            {"episode": "frozen", "role": "frozen_same_source_holdout", "video_path": "c.mp4"},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    entries = load_video_entries(path)

    assert [entry["episode"] for entry in entries] == ["dev", "shift"]


def test_geometrically_implausible_rows_are_removed() -> None:
    rows = (
        row(0.15, 0.40),
        row(0.40, 0.47),
        row(0.60, 0.53),
        row(1.25, 0.70),
        row(0.55, -0.30),
    )

    kept = filter_plausible_rows(rows)

    assert [item.near_x_norm for item in kept] == [0.15, 0.40, 0.60]


def test_optical_flow_keeps_frozen_pixel_scale_for_larger_video_frames() -> None:
    canonical = np.zeros((360, 640, 3), dtype=np.uint8)
    source_native = np.zeros((180, 320, 3), dtype=np.uint8)

    assert prepare_optical_flow_frame(canonical).shape == (180, 320, 3)
    assert prepare_optical_flow_frame(source_native) is source_native


def test_ordered_tracking_preserves_ids_while_rows_move() -> None:
    tracker = TemporalCorridorTracker(TemporalConfig(confirm_frames=2))
    first = tracker.update(corridor_rows())
    second = tracker.update(corridor_rows(0.015))

    assert first.status == "candidate"
    assert second.status == "valid"
    assert second.track_ids == first.track_ids
    assert list(second.track_ids) == sorted(second.track_ids)


def test_one_missing_boundary_can_never_be_valid() -> None:
    tracker = TemporalCorridorTracker(TemporalConfig(confirm_frames=2, max_missing_frames=3))
    tracker.update(corridor_rows())
    valid = tracker.update(corridor_rows())
    missing_right = tracker.update((row(0.18, 0.38), row(0.38, 0.46)))

    assert valid.status == "valid"
    assert missing_right.status == "degraded"
    assert missing_right.navigation_available is False
    assert missing_right.corridor_center_near_x_norm is None


def test_short_occlusion_recovers_same_corridor_identity() -> None:
    tracker = TemporalCorridorTracker(TemporalConfig(confirm_frames=2, recovery_frames=2))
    tracker.update(corridor_rows())
    before = tracker.update(corridor_rows())
    tracker.update(())
    first_return = tracker.update(corridor_rows(0.005))
    recovered = tracker.update(corridor_rows(0.008))

    assert before.status == "valid"
    assert first_return.status == "candidate"
    assert recovered.status == "valid"
    assert recovered.corridor_track_ids == before.corridor_track_ids


def test_same_candidate_can_confirm_across_one_degraded_frame() -> None:
    tracker = TemporalCorridorTracker(
        TemporalConfig(confirm_frames=2, confirmation_window_frames=3)
    )

    first = tracker.update(corridor_rows())
    gap = tracker.update(())
    confirmed = tracker.update(corridor_rows(0.004))

    assert first.status == "candidate"
    assert gap.status == "degraded"
    assert gap.navigation_available is False
    assert confirmed.status == "valid"


def test_central_crop_row_blocks_navigation() -> None:
    tracker = TemporalCorridorTracker(TemporalConfig(confirm_frames=2))
    central = (row(0.20, 0.40), row(0.50, 0.50), row(0.80, 0.60))

    tracker.update(central)
    result = tracker.update(central)

    assert result.status == "degraded"
    assert result.navigation_available is False
    assert "central" in result.reason


def test_new_corridor_requires_consecutive_confirmation() -> None:
    tracker = TemporalCorridorTracker(
        TemporalConfig(confirm_frames=2, switch_confirm_frames=3, association_gate=0.10)
    )
    tracker.update(corridor_rows())
    original = tracker.update(corridor_rows())
    shifted = tuple(row(r.near_x_norm + 0.22, r.far_x_norm + 0.14) for r in corridor_rows())

    first = tracker.update(shifted)
    second = tracker.update(shifted)
    switched = tracker.update(shifted)

    assert original.status == "valid"
    assert first.status != "valid"
    assert second.status != "valid"
    assert switched.status == "valid"
    assert switched.corridor_track_ids != original.corridor_track_ids


def test_geometrically_same_corridor_reidentifies_after_tracks_expire() -> None:
    tracker = TemporalCorridorTracker(
        TemporalConfig(
            confirm_frames=3,
            recovery_frames=2,
            max_missing_frames=1,
            switch_confirm_frames=4,
        )
    )
    tracker.update(corridor_rows())
    tracker.update(corridor_rows())
    before = tracker.update(corridor_rows())
    tracker.update(())
    expired = tracker.update(())
    first_return = tracker.update(corridor_rows(0.004))
    recovered = tracker.update(corridor_rows(0.006))

    assert before.status == "valid"
    assert expired.status == "reject"
    assert first_return.status == "candidate"
    assert recovered.status == "valid"
    assert recovered.corridor_track_ids != before.corridor_track_ids


def test_confidence_weighted_filter_reduces_synthetic_jitter() -> None:
    rng = np.random.default_rng(65)
    tracker = TemporalCorridorTracker(TemporalConfig(confirm_frames=2))
    raw_centers: list[float] = []
    filtered_centers: list[float] = []
    for noise in rng.normal(0.0, 0.018, 80):
        rows = corridor_rows(float(noise))
        raw_centers.append((rows[1].near_x_norm + rows[2].near_x_norm) / 2.0)
        result = tracker.update(rows)
        if result.navigation_available:
            filtered_centers.append(float(result.corridor_center_near_x_norm))

    assert temporal_jitter(filtered_centers) <= 0.70 * temporal_jitter(raw_centers[1:])


def test_rejects_after_missing_tracks_expire() -> None:
    tracker = TemporalCorridorTracker(
        TemporalConfig(confirm_frames=2, max_missing_frames=2)
    )
    tracker.update(corridor_rows())
    tracker.update(corridor_rows())
    tracker.update(())
    tracker.update(())
    result = tracker.update(())

    assert result.status == "reject"
    assert result.navigation_available is False


def test_summary_counts_status_and_corridor_switches() -> None:
    records = [
        {"raw_status": "valid", "temporal_status": "candidate", "raw_center": 0.50,
         "temporal_center": None, "raw_pair": [0, 1], "temporal_pair": None},
        {"raw_status": "valid", "temporal_status": "valid", "raw_center": 0.54,
         "temporal_center": 0.51, "raw_pair": [0, 1], "temporal_pair": [4, 5]},
        {"raw_status": "degraded", "temporal_status": "degraded", "raw_center": None,
         "temporal_center": None, "raw_pair": None, "temporal_pair": [4, 5]},
        {"raw_status": "valid", "temporal_status": "valid", "raw_center": 0.52,
         "temporal_center": 0.515, "raw_pair": [1, 2], "temporal_pair": [6, 7]},
    ]

    summary = summarize_temporal_records(records)

    assert summary["frame_count"] == 4
    assert summary["raw_valid_fraction"] == 0.75
    assert summary["temporal_valid_fraction"] == 0.5
    assert summary["raw_status_switches"] == 2
    assert summary["temporal_corridor_switches"] == 1


def test_summary_does_not_call_gaps_or_episode_boundaries_jitter() -> None:
    records = [
        {"episode": "a", "raw_status": "valid", "temporal_status": "valid",
         "raw_center": 0.40, "temporal_center": 0.45, "raw_pair": [0, 1], "temporal_pair": [2, 3]},
        {"episode": "a", "raw_status": "degraded", "temporal_status": "degraded",
         "raw_center": None, "temporal_center": None, "raw_pair": None, "temporal_pair": None},
        {"episode": "a", "raw_status": "valid", "temporal_status": "valid",
         "raw_center": 0.70, "temporal_center": 0.65, "raw_pair": [1, 2], "temporal_pair": [4, 5]},
        {"episode": "b", "raw_status": "valid", "temporal_status": "valid",
         "raw_center": 0.10, "temporal_center": 0.15, "raw_pair": [0, 1], "temporal_pair": [0, 1]},
    ]

    summary = summarize_temporal_records(records)

    assert summary["raw_center_jitter_norm"] == 0.0
    assert summary["temporal_center_jitter_norm"] == 0.0
    assert summary["raw_status_switches"] == 2
    assert summary["raw_corridor_switches"] == 1


def test_synthetic_acceptance_exercises_safety_and_jitter() -> None:
    result = run_synthetic_acceptance_study(
        TemporalConfig(confirm_frames=3, recovery_frames=2, max_missing_frames=3)
    )

    assert result["unsupported_frame_count"] > 0
    assert result["unsafe_false_valid_rate"] == 0.0
    assert result["one_side_missing_valid_count"] == 0
    assert result["central_row_valid_count"] == 0
    assert result["jitter_reduction_fraction"] >= 0.30


def test_temporal_acceptance_requires_stability_safety_and_retained_coverage() -> None:
    aggregate = {
        "raw_valid_fraction": 0.13,
        "temporal_valid_fraction": 0.045,
        "raw_status_switches": 5000,
        "temporal_status_switches": 600,
        "raw_corridor_switches": 130,
        "temporal_corridor_switches": 65,
        "raw_center_jitter_norm": 0.047,
        "temporal_center_jitter_norm": 0.008,
    }
    synthetic = {
        "unsafe_false_valid_rate": 0.0,
        "one_side_missing_valid_count": 0,
        "central_row_valid_count": 0,
    }

    checks = temporal_acceptance_checks(aggregate, synthetic)

    assert all(checks.values())


def test_tiny_video_episode_runs_every_frame_without_using_labels(tmp_path: Path) -> None:
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
    )
    assert writer.isOpened()
    for value in range(8):
        writer.write(np.full((64, 96, 3), value * 20, dtype=np.uint8))
    writer.release()

    class TinyPredictor:
        def predict(self, frames: list[np.ndarray]) -> list[tuple[CropRowLine, ...]]:
            return [corridor_rows(index * 0.002) for index in range(len(frames))]

    records, summary = process_video_episode(
        video,
        predictor=TinyPredictor(),
        config=TemporalConfig(confirm_frames=2),
        episode="tiny",
        use_optical_flow=True,
    )

    assert len(records) == 8
    assert summary["frame_count"] == 8
    assert records[0]["episode"] == "tiny"
    assert records[-1]["frame_index"] == 7
    assert summary["decode_complete"] is True
    assert summary["optical_flow_enabled"] is True
    assert records[0]["flow_only_row_count"] == 0
    assert len(records[0]["plausible_rows"]) == 4
    assert set(records[0]["plausible_rows"][0]) == {
        "near_x_norm", "far_x_norm", "confidence", "support_band_count"
    }
    assert len(records[0]["temporal_observation_rows"]) == 4
    assert len(records[0]["temporal_rows"]) == 4


def test_video_feature_reuses_frozen_four_channel_contract() -> None:
    image = np.zeros((90, 160, 3), dtype=np.uint8)
    image[:, 60:100, 1] = 180

    feature = prepare_video_feature(image, resolution=192)

    assert feature.shape == (4, 192, 192)
    assert feature.dtype == np.uint8
    assert set(np.unique(feature[3])).issubset({0, 255})


def test_study_writes_development_results_without_opening_frozen_video(
    tmp_path: Path,
) -> None:
    video = tmp_path / "dev.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
    )
    assert writer.isOpened()
    for value in range(6):
        writer.write(np.full((64, 96, 3), value * 15, dtype=np.uint8))
    writer.release()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode": "dev", "role": "temporal_development", "video_path": str(video)},
                    {"episode": "frozen", "role": "frozen_same_source_holdout",
                     "video_path": str(tmp_path / "must_not_open.mp4")},
                ]
            }
        ),
        encoding="utf-8",
    )

    class TinyPredictor:
        def predict(self, frames: list[np.ndarray]) -> list[tuple[CropRowLine, ...]]:
            return [corridor_rows() for _ in frames]

    output = tmp_path / "output"
    result = run_day65_video_study(
        manifest_path=manifest,
        output_dir=output,
        predictor=TinyPredictor(),
        config=TemporalConfig(confirm_frames=2),
    )

    assert result["episode_count"] == 1
    assert result["frozen_video_frames_accessed"] is False
    assert result["all_videos_decode_complete"] is True
    assert (output / "day65_results.json").is_file()
    assert (output / "dev.jsonl").is_file()


def test_frozen_predictor_rejects_missing_checkpoint(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"

    try:
        FrozenDay63Predictor(missing, device="cpu")
    except ValueError as error:
        assert "checkpoint" in str(error)
    else:
        raise AssertionError("missing frozen checkpoint must be rejected")


def test_temporal_overlay_preserves_shape_and_draws_status() -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    record = {
        "raw_status": "valid",
        "temporal_status": "valid",
        "plausible_rows": [
            {"near_x_norm": 0.35, "far_x_norm": 0.46, "confidence": 0.8,
             "support_band_count": 8},
            {"near_x_norm": 0.65, "far_x_norm": 0.54, "confidence": 0.8,
             "support_band_count": 8},
        ],
        "temporal_rows": [
            {"near_x_norm": 0.36, "far_x_norm": 0.46, "confidence": 0.8,
             "support_band_count": 8},
            {"near_x_norm": 0.64, "far_x_norm": 0.54, "confidence": 0.8,
             "support_band_count": 8},
        ],
        "temporal_center": 0.50,
        "temporal_heading": 0.0,
        "flow_only_row_count": 1,
    }

    overlay = draw_temporal_overlay(image, record)

    assert overlay.shape == image.shape
    assert np.count_nonzero(overlay) > 0


def test_optical_flow_propagates_row_with_forward_backward_check() -> None:
    rng = np.random.default_rng(6502)
    previous = rng.integers(0, 90, size=(120, 180, 3), dtype=np.uint8)
    cv2.line(previous, (54, 108), (72, 48), (0, 240, 0), 7)
    transform = np.float32([[1, 0, 5], [0, 1, 0]])
    current = cv2.warpAffine(previous, transform, (180, 120))
    source = row(54 / 179, 72 / 179)

    propagated = propagate_rows_optical_flow(previous, current, (source,))

    assert len(propagated) == 1
    assert propagated[0].near_x_norm == pytest.approx(59 / 179, abs=0.025)
    assert propagated[0].far_x_norm == pytest.approx(77 / 179, abs=0.025)
    assert propagated[0].confidence < source.confidence


def test_optical_flow_rejects_featureless_transition() -> None:
    previous = np.zeros((100, 160, 3), dtype=np.uint8)
    current = np.zeros_like(previous)

    propagated = propagate_rows_optical_flow(
        previous, current, (row(0.35, 0.45),)
    )

    assert propagated == ()


def test_optical_flow_bridge_only_fills_a_short_detection_gap() -> None:
    rng = np.random.default_rng(6503)
    first = rng.integers(0, 100, size=(120, 180, 3), dtype=np.uint8)
    cv2.line(first, (54, 108), (72, 48), (0, 240, 0), 7)
    frames = [
        cv2.warpAffine(first, np.float32([[1, 0, shift], [0, 1, 0]]), (180, 120))
        for shift in (0, 3, 6, 9)
    ]
    bridge = OpticalFlowObservationBridge(max_flow_age=2)

    seeded, seed_flow = bridge.update(frames[0], (row(54 / 179, 72 / 179),))
    filled_once, flow_once = bridge.update(frames[1], ())
    filled_twice, flow_twice = bridge.update(frames[2], ())
    expired, flow_expired = bridge.update(frames[3], ())

    assert len(seeded) == 1 and seed_flow == 0
    assert len(filled_once) == 1 and flow_once == 1
    assert len(filled_twice) == 1 and flow_twice == 1
    assert expired == () and flow_expired == 0
