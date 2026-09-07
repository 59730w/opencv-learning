from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
DAY63_CODE = ROOT / "63_crop_row_geometry_extraction" / "code"
DAY64_CODE = ROOT / "64_crop_row_camera_coordinates_measurement" / "code"
DAY65_CODE = ROOT / "65_crop_row_video_temporal_stability" / "code"
DAY66_CODE = ROOT / "66_crop_row_offline_video_pilot" / "code"
for dependency in (DAY63_CODE, DAY64_CODE, DAY65_CODE, DAY66_CODE):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from day66_offline_video_pilot import (  # noqa: E402
    _wrap_overlay_text,
    benchmark_canonical_cpu,
    build_execution_predictor,
    build_visual_audit,
    build_arg_parser,
    draw_pilot_overlay,
    load_and_verify_frozen_config,
    OptimizedCpuFrozenDay63Predictor,
    pilot_acceptance_checks,
    prepare_optical_flow_frame,
    project_navigation_contract,
    run_day66_pilot,
    run_episode_pilot,
    select_stratified_audit_samples,
    verify_rendered_video,
)
from day63_crop_row_geometry import CropRowLine, ResNet18RowUNet  # noqa: E402
from day65_video_temporal import FrozenDay63Predictor, TemporalConfig  # noqa: E402


SELECTED_CONFIG = {
    "association_gate": 0.25,
    "max_missing_frames": 8,
    "confirm_frames": 2,
    "recovery_frames": 1,
    "switch_confirm_frames": 6,
    "confirmation_window_frames": 8,
    "process_variance": 1e-5,
    "measurement_variance": 0.0025,
    "minimum_row_confidence": 0.12,
}


def write_frozen_files(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "frozen.json"
    result_path = tmp_path / "day65_results.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_config": SELECTED_CONFIG,
                "optical_flow_enabled": True,
                "allowed_roles": ["temporal_development", "shifted_development"],
            }
        ),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "marker": "DAY65_TEMPORAL_DEVELOPMENT_COMPLETE",
                "config": SELECTED_CONFIG,
                "optical_flow_enabled": True,
                "frozen_video_frames_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return config_path, result_path


def test_frozen_config_exactly_matches_accepted_day65_result(tmp_path: Path) -> None:
    config_path, result_path = write_frozen_files(tmp_path)

    config, optical_flow_enabled, evidence = load_and_verify_frozen_config(
        config_path, result_path
    )

    assert config.__dict__ == SELECTED_CONFIG
    assert optical_flow_enabled is True
    assert evidence["exact_day65_match"] is True
    assert len(evidence["frozen_config_sha256"]) == 64
    assert len(evidence["day65_result_sha256"]) == 64


def test_frozen_config_rejects_any_parameter_drift(tmp_path: Path) -> None:
    config_path, result_path = write_frozen_files(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["selected_config"]["confirm_frames"] = 3
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exactly match"):
        load_and_verify_frozen_config(config_path, result_path)


def test_candidate_is_publicly_degraded_and_never_navigation_available() -> None:
    projected = project_navigation_contract(
        {
            "episode": "dev",
            "frame_index": 4,
            "temporal_status": "candidate",
            "temporal_center": None,
            "temporal_heading": None,
            "temporal_vanishing_point": None,
            "temporal_pair": [7, 8],
            "temporal_track_ids": [7, 8],
            "temporal_rows": [
                {"near_x_norm": 0.35, "far_x_norm": 0.45, "confidence": 0.8,
                 "support_band_count": 8},
                {"near_x_norm": 0.65, "far_x_norm": 0.55, "confidence": 0.7,
                 "support_band_count": 8},
            ],
            "temporal_confidence": 0.75,
            "temporal_reason": "awaiting confirmation",
        }
    )

    assert projected["pilot_state"] == "candidate"
    assert projected["status"] == "degraded"
    assert projected["navigation_available"] is False
    for field in (
        "corridor_left_boundary",
        "corridor_right_boundary",
        "corridor_centerline_points_norm",
        "lateral_offset_norm",
        "heading_error_deg",
        "vanishing_point_norm",
        "row_spacing_norm",
    ):
        assert projected[field] is None


def test_valid_projection_emits_contract_fields_from_selected_track_ids() -> None:
    projected = project_navigation_contract(
        {
            "episode": "dev",
            "frame_index": 5,
            "temporal_status": "valid",
            "temporal_center": 0.52,
            "temporal_heading": 2.0,
            "temporal_vanishing_point": [0.51, 0.12],
            "temporal_pair": [11, 12],
            "temporal_track_ids": [10, 11, 12, 13],
            "temporal_rows": [
                {"near_x_norm": 0.15, "far_x_norm": 0.35, "confidence": 0.6,
                 "support_band_count": 8},
                {"near_x_norm": 0.38, "far_x_norm": 0.46, "confidence": 0.9,
                 "support_band_count": 8},
                {"near_x_norm": 0.66, "far_x_norm": 0.54, "confidence": 0.8,
                 "support_band_count": 8},
                {"near_x_norm": 0.84, "far_x_norm": 0.65, "confidence": 0.5,
                 "support_band_count": 8},
            ],
            "temporal_confidence": 0.85,
            "temporal_reason": "confirmed",
        }
    )

    assert projected["pilot_state"] == "valid"
    assert projected["status"] == "valid"
    assert projected["navigation_available"] is True
    assert projected["corridor_left_boundary"]["track_id"] == 11
    assert projected["corridor_right_boundary"]["track_id"] == 12
    assert projected["corridor_centerline_points_norm"] == [
        [pytest.approx(0.52), 0.90],
        [pytest.approx(0.50), 0.40],
    ]
    assert projected["lateral_offset_norm"] == pytest.approx(0.02)
    assert projected["heading_error_deg"] == pytest.approx(2.0)
    assert projected["row_spacing_norm"] == pytest.approx(0.28)
    assert projected["confidence"] == pytest.approx(0.85)


def pilot_overlay_record(state: str) -> dict:
    valid = state == "valid"
    return {
        "episode": "episode",
        "frame_index": 7,
        "crop_rows": [
            {"track_id": 11, "near_x_norm": 0.38, "far_x_norm": 0.46,
             "confidence": 0.9, "support_band_count": 8, "status": "tracked"},
            {"track_id": 12, "near_x_norm": 0.66, "far_x_norm": 0.54,
             "confidence": 0.8, "support_band_count": 8, "status": "tracked"},
        ],
        "corridor_left_boundary": {
            "track_id": 11, "near_x_norm": 0.38, "far_x_norm": 0.46,
            "confidence": 0.9,
        } if valid else None,
        "corridor_right_boundary": {
            "track_id": 12, "near_x_norm": 0.66, "far_x_norm": 0.54,
            "confidence": 0.8,
        } if valid else None,
        "corridor_centerline_points_norm": [[0.52, 0.90], [0.50, 0.40]] if valid else None,
        "lateral_offset_norm": 0.02 if valid else None,
        "heading_error_deg": 2.0 if valid else None,
        "vanishing_point_norm": [0.51, 0.12] if valid else None,
        "row_spacing_norm": 0.28 if valid else None,
        "confidence": 0.85,
        "pilot_state": state,
        "status": "valid" if valid else ("reject" if state == "reject" else "degraded"),
        "navigation_available": valid,
        "reason": "test reason",
        "diagnostics": {"flow_only_row_count": 0},
    }


def test_overlay_draws_tracks_selected_boundaries_and_valid_center() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    overlay = draw_pilot_overlay(frame, pilot_overlay_record("valid"))

    assert overlay.shape == frame.shape
    assert np.count_nonzero(overlay) > 500
    middle = overlay[105:130, 160:174]
    green_pixels = (middle[:, :, 1] > 180) & (middle[:, :, 0] < 80) & (middle[:, :, 2] < 80)
    assert int(green_pixels.sum()) > 5
    assert not np.array_equal(overlay, frame)


@pytest.mark.parametrize("state", ["candidate", "degraded", "reject"])
def test_overlay_never_draws_navigation_center_outside_valid(state: str) -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    overlay = draw_pilot_overlay(frame, pilot_overlay_record(state))

    middle = overlay[105:130, 160:174]
    green_pixels = (middle[:, :, 1] > 180) & (middle[:, :, 0] < 80) & (middle[:, :, 2] < 80)
    assert int(green_pixels.sum()) == 0


def test_overlay_marks_vanishing_point_only_for_valid_navigation() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    valid = draw_pilot_overlay(frame, pilot_overlay_record("valid"))
    candidate = draw_pilot_overlay(frame, pilot_overlay_record("candidate"))

    vp = (round(0.51 * 319), round(0.12 * 179))
    assert cv2.countNonZero(cv2.cvtColor(valid[max(0, vp[1]-5):vp[1]+6, vp[0]-5:vp[0]+6], cv2.COLOR_BGR2GRAY)) > 0
    assert cv2.countNonZero(cv2.cvtColor(candidate[max(0, vp[1]-5):vp[1]+6, vp[0]-5:vp[0]+6], cv2.COLOR_BGR2GRAY)) == 0


def test_valid_overlay_explicitly_labels_boundaries_heading_and_direction_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered_text: list[str] = []
    original_put_text = cv2.putText

    def capture_text(image: np.ndarray, text: str, *args: object, **kwargs: object) -> np.ndarray:
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    overlay = draw_pilot_overlay(frame, pilot_overlay_record("valid"))

    assert "LEFT=ID11" in rendered_text
    assert "RIGHT=ID12" in rendered_text
    assert any("heading=+2.0deg" in text for text in rendered_text)
    yellow = (
        (overlay[:, :, 0] < 80)
        & (overlay[:, :, 1] > 180)
        & (overlay[:, :, 2] > 180)
    )
    assert int(yellow.sum()) > 5


def test_selected_boundary_labels_stay_below_direction_hud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    text_positions: dict[str, tuple[int, int]] = {}
    original_put_text = cv2.putText

    def capture_text(
        image: np.ndarray, text: str, origin: tuple[int, int],
        *args: object, **kwargs: object,
    ) -> np.ndarray:
        text_positions[text] = origin
        return original_put_text(image, text, origin, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    draw_pilot_overlay(frame, pilot_overlay_record("valid"))

    assert text_positions["LEFT=ID11"][1] >= round(0.55 * frame.shape[0])
    assert text_positions["RIGHT=ID12"][1] >= round(0.55 * frame.shape[0])
    assert text_positions["DIR"][1] < text_positions["LEFT=ID11"][1]


def test_overlay_text_keeps_dark_outline_on_bright_background() -> None:
    frame = np.full((360, 640, 3), 255, dtype=np.uint8)

    overlay = draw_pilot_overlay(frame, pilot_overlay_record("candidate"))

    secondary_hud = overlay[32:58, 0:500]
    dark_pixels = np.all(secondary_hud < 80, axis=2)
    assert int(dark_pixels.sum()) > 40


@pytest.mark.parametrize("state", ["candidate", "degraded", "reject"])
def test_nonvalid_overlay_marks_heading_unavailable_without_direction_proxy(
    state: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered_text: list[str] = []
    original_put_text = cv2.putText

    def capture_text(image: np.ndarray, text: str, *args: object, **kwargs: object) -> np.ndarray:
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    overlay = draw_pilot_overlay(frame, pilot_overlay_record(state))

    assert any("heading=n/a" in text for text in rendered_text)
    direction_hud = overlay[55:115, 500:625]
    yellow = (
        (direction_hud[:, :, 0] < 80)
        & (direction_hud[:, :, 1] > 180)
        & (direction_hud[:, :, 2] > 180)
    )
    assert int(yellow.sum()) == 0


def test_overlay_reason_wrapper_keeps_each_line_inside_frame() -> None:
    text = "both supported adjacent crop-row boundaries are required before navigation"

    lines = _wrap_overlay_text(text, max_width=620, font_scale=0.60, thickness=2)

    assert 1 < len(lines) <= 2
    assert " ".join(lines) == text
    assert all(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0][0] <= 620
        for line in lines
    )


class StableRowsPredictor:
    def predict(self, frames: list[np.ndarray]) -> list[tuple[CropRowLine, ...]]:
        rows = (
            CropRowLine(0.18, 0.38, 0.92, 8),
            CropRowLine(0.38, 0.46, 0.91, 8),
            CropRowLine(0.62, 0.54, 0.90, 8),
            CropRowLine(0.82, 0.62, 0.89, 8),
        )
        return [rows for _ in frames]


def write_tiny_video(path: Path, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((64, 96, 3), 30 + index * 20, dtype=np.uint8)
        cv2.putText(frame, str(index), (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_episode_pilot_writes_frame_aligned_jsonl_and_readable_overlay(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_tiny_video(source, frame_count=4)

    records, summary = run_episode_pilot(
        {"episode": "dev_ep", "role": "temporal_development", "video_path": str(source)},
        predictor=StableRowsPredictor(),
        config=TemporalConfig(confirm_frames=2),
        use_optical_flow=False,
        output_dir=tmp_path / "output",
    )

    written = read_jsonl(tmp_path / "output" / "dev_ep.jsonl")
    overlay = tmp_path / "output" / "dev_ep_day66_overlay.mp4"
    assert len(records) == len(written) == 4
    assert [row["frame_index"] for row in written] == [0, 1, 2, 3]
    assert records[0]["pilot_state"] == "candidate"
    assert records[1]["pilot_state"] == "valid"
    assert summary["source_frame_count"] == 4
    assert summary["rendered_frame_count"] == 4
    assert summary["jsonl_record_count"] == 4
    assert summary["overlay_decode_complete"] is True
    rendered = verify_rendered_video(overlay, expected_frames=4)
    assert rendered["passed"] is True
    assert rendered["width"] == 192
    assert rendered["height"] == 128
    assert summary["render_scale"] == 2.0
    assert len(summary["overlay_sha256"]) == 64
    assert len(summary["jsonl_sha256"]) == 64


def test_each_episode_starts_with_fresh_temporal_state(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    write_tiny_video(first, frame_count=2)
    write_tiny_video(second, frame_count=2)
    config = TemporalConfig(confirm_frames=2)

    first_records, _ = run_episode_pilot(
        {"episode": "first", "role": "temporal_development", "video_path": str(first)},
        predictor=StableRowsPredictor(), config=config, use_optical_flow=False,
        output_dir=tmp_path / "output",
    )
    second_records, _ = run_episode_pilot(
        {"episode": "second", "role": "shifted_development", "video_path": str(second)},
        predictor=StableRowsPredictor(), config=config, use_optical_flow=False,
        output_dir=tmp_path / "output",
    )

    assert first_records[0]["pilot_state"] == "candidate"
    assert second_records[0]["pilot_state"] == "candidate"


def test_full_runner_structurally_excludes_missing_frozen_video(tmp_path: Path) -> None:
    source = tmp_path / "dev.mp4"
    write_tiny_video(source, frame_count=3)
    config_path, result_path = write_frozen_files(tmp_path)
    manifest = {
        "episodes": [
            {"episode": "dev", "role": "temporal_development", "video_path": str(source)},
            {"episode": "frozen", "role": "frozen_same_source_holdout",
             "video_path": str(tmp_path / "must_not_be_opened.mp4")},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_day66_pilot(
        manifest_path=manifest_path,
        frozen_config_path=config_path,
        day65_result_path=result_path,
        checkpoint_path=tmp_path / "model.pt",
        output_dir=tmp_path / "pilot",
        predictor=StableRowsPredictor(),
        device="cpu",
    )

    assert result["marker"] == "DAY66_OFFLINE_PILOT_COMPLETE"
    assert result["episode_count"] == 1
    assert result["aggregate"]["frame_count"] == 3
    assert result["roles"] == ["temporal_development"]
    assert result["frozen_video_frames_accessed"] is False
    assert result["frozen_role_structurally_excluded"] is True
    assert all(result["acceptance_checks"].values())
    assert set(result["implementation_sha256"]) == {
        "day61", "day62", "day63", "day65", "day66"
    }
    assert not (tmp_path / "pilot" / "frozen.jsonl").exists()

    audit = build_visual_audit(tmp_path / "pilot", per_group=1, columns=2)
    assert audit["record_count"] == 3
    assert audit["selected_counts"]["candidate"] == 1
    assert audit["selected_counts"]["valid"] == 1
    assert Path(audit["contact_sheets"]["candidate"]).is_file()
    assert Path(audit["contact_sheets"]["valid"]).is_file()
    assert (tmp_path / "pilot" / "day66_visual_audit.json").is_file()


def test_render_verifier_detects_frame_count_mismatch(tmp_path: Path) -> None:
    video = tmp_path / "rendered.mp4"
    write_tiny_video(video, frame_count=2)

    check = verify_rendered_video(video, expected_frames=3)

    assert check["passed"] is False
    assert check["decoded_frame_count"] == 2


def test_acceptance_rejects_navigation_payload_on_nonvalid_frame() -> None:
    result = {
        "episode_count": 1,
        "aggregate": {"frame_count": 1},
        "episode_summaries": [
            {"decode_complete": True, "overlay_decode_complete": True,
             "source_frame_count": 1, "rendered_frame_count": 1,
             "jsonl_record_count": 1}
        ],
        "roles": ["temporal_development"],
        "frozen_video_frames_accessed": False,
        "frozen_role_structurally_excluded": True,
        "frozen_config_evidence": {"exact_day65_match": True},
        "navigation_invariant_violations": 1,
    }

    checks = pilot_acceptance_checks(result)

    assert checks["navigation_outputs_only_when_valid"] is False


def test_cpu_benchmark_declares_boundary_and_excludes_warmup(tmp_path: Path) -> None:
    video = tmp_path / "benchmark.mp4"
    write_tiny_video(video, frame_count=5)

    result = benchmark_canonical_cpu(
        video_path=video,
        checkpoint_path=tmp_path / "unused.pt",
        config=TemporalConfig(confirm_frames=2),
        use_optical_flow=False,
        max_frames=4,
        warmup_frames=2,
        cpu_threads=1,
        predictor=StableRowsPredictor(),
    )

    assert result["device"] == "cpu"
    assert result["canonical_size"] == [640, 360]
    assert result["warmup_frames"] == 2
    assert result["measured_frames"] == 4
    assert result["timing_sample_count"] == 4
    assert result["decode_resize_inside_timing"] is False
    assert result["overlay_encode_inside_timing"] is False
    assert result["optical_flow_max_working_size"] == [320, 180]
    assert result["timing_boundary"] == (
        "single-frame frozen-model inference plus plausibility filtering, "
        "optical-flow bridge and temporal tracking"
    )
    assert result["median_ms"] >= 0.0
    assert result["p95_ms"] >= result["median_ms"]
    assert isinstance(result["target_median_at_most_50ms_passed"], bool)


def test_optical_flow_working_frame_downscales_canonical_but_never_upscales() -> None:
    canonical = np.zeros((360, 640, 3), dtype=np.uint8)
    small = np.zeros((90, 160, 3), dtype=np.uint8)

    canonical_flow = prepare_optical_flow_frame(canonical)
    small_flow = prepare_optical_flow_frame(small)

    assert canonical_flow.shape == (180, 320, 3)
    assert small_flow.shape == small.shape


def test_stratified_audit_selection_covers_states_transitions_and_flow() -> None:
    records = []
    states = ["candidate", "valid", "valid", "degraded", "reject", "valid"]
    for index, state in enumerate(states):
        records.append(
            {
                "episode": "ep",
                "frame_index": index,
                "pilot_state": state,
                "diagnostics": {"flow_only_row_count": 1 if index in {2, 3} else 0},
            }
        )

    selected = select_stratified_audit_samples(records, per_group=2)

    assert set(("valid", "candidate", "degraded", "reject")).issubset(selected)
    assert len(selected["valid"]) == 2
    assert [row["frame_index"] for row in selected["candidate"]] == [0]
    assert [row["frame_index"] for row in selected["reject"]] == [4]
    assert len(selected["state_transition"]) == 2
    assert [row["frame_index"] for row in selected["flow_bridge"]] == [2, 3]


def test_cli_does_not_expose_day65_tuning_parameters() -> None:
    parser = build_arg_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    forbidden = {
        "--association-gate",
        "--max-missing-frames",
        "--confirm-frames",
        "--recovery-frames",
        "--switch-confirm-frames",
        "--confirmation-window-frames",
        "--process-variance",
        "--measurement-variance",
        "--minimum-row-confidence",
        "--no-optical-flow",
    }
    assert option_strings.isdisjoint(forbidden)


def test_optimized_cpu_predictor_preserves_eager_decoded_rows(tmp_path: Path) -> None:
    torch.manual_seed(66)
    model = ResNet18RowUNet()
    checkpoint = tmp_path / "tiny_model.pt"
    torch.save(
        {
            "input": "RGB_plus_frozen_Day62_mask",
            "resolution": 32,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    rng = np.random.default_rng(66)
    frames = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(2)]
    eager = FrozenDay63Predictor(checkpoint, device="cpu", batch_size=1)

    optimized = OptimizedCpuFrozenDay63Predictor(checkpoint)

    eager_rows = eager.predict(frames)
    optimized_rows = optimized.predict(frames)
    assert optimized.backend == "torchscript_optimize_for_inference_channels_last"
    assert len(optimized_rows) == len(eager_rows) == 2
    assert [len(rows) for rows in optimized_rows] == [len(rows) for rows in eager_rows]
    for eager_frame, optimized_frame in zip(eager_rows, optimized_rows):
        for expected, actual in zip(eager_frame, optimized_frame):
            assert actual.near_x_norm == pytest.approx(expected.near_x_norm, abs=1e-5)
            assert actual.far_x_norm == pytest.approx(expected.far_x_norm, abs=1e-5)
            assert actual.confidence == pytest.approx(expected.confidence, abs=1e-5)

    selected = build_execution_predictor(checkpoint, device="cpu", batch_size=8)
    assert isinstance(selected, OptimizedCpuFrozenDay63Predictor)
    assert selected.batch_size == 1
