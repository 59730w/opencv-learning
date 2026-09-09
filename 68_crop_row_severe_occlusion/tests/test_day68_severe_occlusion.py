from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from day68_severe_occlusion import (  # noqa: E402
    NAVIGATION_FIELDS,
    OcclusionGuardConfig,
    OcclusionRecoveryGuard,
    apply_guard_to_records,
    assert_manifest_excludes_frozen,
    draw_day68_overlay,
    run_day68,
    summarize_day68,
)


def valid_record(frame_index: int, *, min_support: int = 6) -> dict:
    rows = [
        {
            "track_id": 4,
            "near_x_norm": 0.25,
            "far_x_norm": 0.42,
            "confidence": 0.82,
            "support_band_count": min_support,
            "status": "tracked",
        },
        {
            "track_id": 7,
            "near_x_norm": 0.76,
            "far_x_norm": 0.58,
            "confidence": 0.79,
            "support_band_count": min_support + 2,
            "status": "tracked",
        },
    ]
    return {
        "episode": "development_ep",
        "frame_index": frame_index,
        "crop_rows": rows,
        "corridor_left_boundary": copy.deepcopy(rows[0]),
        "corridor_right_boundary": copy.deepcopy(rows[1]),
        "corridor_centerline_points_norm": [[0.505, 0.90], [0.50, 0.40]],
        "lateral_offset_norm": 0.005,
        "heading_error_deg": -0.5,
        "vanishing_point_norm": [0.50, -0.2],
        "row_spacing_norm": 0.51,
        "confidence": 0.79,
        "pilot_state": "valid",
        "status": "valid",
        "navigation_available": True,
        "reason": "same observed boundary identities remain confirmed",
        "diagnostics": {
            "raw_status": "valid",
            "raw_row_count": 2,
            "plausible_row_count": 2,
            "flow_only_row_count": 0,
            "active_or_selected_track_ids": [4, 7],
        },
        "role": "temporal_development",
    }


def degraded_record(frame_index: int) -> dict:
    return {
        "episode": "development_ep",
        "frame_index": frame_index,
        "crop_rows": [],
        "corridor_left_boundary": None,
        "corridor_right_boundary": None,
        "corridor_centerline_points_norm": None,
        "lateral_offset_norm": None,
        "heading_error_deg": None,
        "vanishing_point_norm": None,
        "row_spacing_norm": None,
        "confidence": 0.0,
        "pilot_state": "degraded",
        "status": "degraded",
        "navigation_available": False,
        "reason": "both supported adjacent crop-row boundaries are required",
        "diagnostics": {
            "raw_status": "degraded",
            "raw_row_count": 0,
            "plausible_row_count": 0,
            "flow_only_row_count": 0,
            "active_or_selected_track_ids": None,
        },
        "role": "temporal_development",
    }


class OcclusionGuardTests(unittest.TestCase):
    def test_config_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            OcclusionGuardConfig(min_preceding_nonvalid_frames=0)
        with self.assertRaises(ValueError):
            OcclusionGuardConfig(max_min_boundary_support=0)

    def test_weak_reentry_after_long_outage_clears_navigation(self):
        records = [degraded_record(i) for i in range(8)] + [valid_record(8, min_support=6)]
        frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in records]
        guarded = apply_guard_to_records(records, OcclusionGuardConfig(), frames=frames)
        final = guarded[-1]
        self.assertEqual(final["pilot_state"], "degraded")
        self.assertFalse(final["navigation_available"])
        self.assertTrue(final["diagnostics"]["day68_guard_triggered"])
        self.assertEqual(final["diagnostics"]["preceding_nonvalid_frames"], 8)
        self.assertEqual(final["diagnostics"]["minimum_boundary_support"], 6)
        self.assertEqual(final["diagnostics"]["corridor_edge_fraction"], 0.0)
        self.assertIn("severe_occlusion_recovery_guard", final["reason"])
        for field in NAVIGATION_FIELDS:
            self.assertIsNone(final[field], field)
        self.assertEqual(final["crop_rows"], records[-1]["crop_rows"])

    def test_strong_boundary_support_after_long_outage_stays_valid(self):
        records = [degraded_record(i) for i in range(12)] + [valid_record(12, min_support=7)]
        final = apply_guard_to_records(records, OcclusionGuardConfig())[-1]
        self.assertEqual(final["pilot_state"], "valid")
        self.assertTrue(final["navigation_available"])
        self.assertFalse(final["diagnostics"]["day68_guard_triggered"])

    def test_short_outage_with_weak_support_stays_valid(self):
        records = [degraded_record(i) for i in range(7)] + [valid_record(7, min_support=4)]
        final = apply_guard_to_records(records, OcclusionGuardConfig())[-1]
        self.assertEqual(final["pilot_state"], "valid")

    def test_guard_is_a_one_frame_recovery_interlock(self):
        records = [degraded_record(i) for i in range(8)]
        records.extend([valid_record(8, min_support=6), valid_record(9, min_support=6)])
        frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in records]
        guarded = apply_guard_to_records(records, OcclusionGuardConfig(), frames=frames)
        self.assertEqual([row["pilot_state"] for row in guarded[-2:]], ["degraded", "valid"])

    def test_high_edge_corridor_does_not_trigger_weak_reentry(self):
        records = [degraded_record(i) for i in range(8)] + [valid_record(8, min_support=6)]
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for x in range(0, 320, 8):
            cv2.line(frame, (x, 0), (x, 179), (255, 255, 255), 2)
        frames = [np.zeros_like(frame) for _ in range(8)] + [frame]
        final = apply_guard_to_records(records, OcclusionGuardConfig(), frames=frames)[-1]
        self.assertEqual(final["pilot_state"], "valid")
        self.assertGreater(final["diagnostics"]["corridor_edge_fraction"], 0.22)

    def test_out_of_frame_boundary_triggers_even_with_high_edges(self):
        records = [degraded_record(i) for i in range(8)] + [valid_record(8, min_support=6)]
        records[-1]["corridor_right_boundary"]["near_x_norm"] = 1.01
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for x in range(0, 320, 8):
            cv2.line(frame, (x, 0), (x, 179), (255, 255, 255), 2)
        frames = [np.zeros_like(frame) for _ in range(8)] + [frame]
        final = apply_guard_to_records(records, OcclusionGuardConfig(), frames=frames)[-1]
        self.assertEqual(final["pilot_state"], "degraded")
        self.assertTrue(final["diagnostics"]["boundary_out_of_frame"])

    def test_existing_nonvalid_record_remains_navigation_free(self):
        source = degraded_record(0)
        result = OcclusionRecoveryGuard(OcclusionGuardConfig()).apply(source)
        self.assertEqual(result["pilot_state"], "degraded")
        self.assertFalse(result["navigation_available"])
        self.assertFalse(result["diagnostics"]["day68_guard_triggered"])

    def test_inputs_must_exclude_frozen_role(self):
        allowed = {"episodes": [{"episode": "a", "role": "temporal_development"}]}
        self.assertEqual(assert_manifest_excludes_frozen(allowed), 1)
        frozen = {"episodes": [{"episode": "z", "role": "frozen_same_source_holdout"}]}
        with self.assertRaises(ValueError):
            assert_manifest_excludes_frozen(frozen)

    def test_summary_counts_target_reduction_and_coverage_cost(self):
        baseline = [degraded_record(i) for i in range(8)] + [valid_record(8, min_support=6)]
        frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in baseline]
        guarded = apply_guard_to_records(baseline, OcclusionGuardConfig(), frames=frames)
        annotations = [
            {
                "event_id": "development_ep__000008_000008__valid__stable_valid_output",
                "episode": "development_ep",
                "primary_visual_label": "severe_occlusion",
                "navigation_assessment": "suspected_unsafe",
            }
        ]
        summary = summarize_day68(baseline, guarded, annotations, minimum_valid_retention=0.0)
        self.assertEqual(summary["baseline_suspected_unsafe_severe_valid_event_count"], 1)
        self.assertEqual(summary["guarded_suspected_unsafe_severe_valid_event_count"], 0)
        self.assertEqual(summary["navigation_invariant_violations"], 0)
        self.assertEqual(summary["guard_triggered_review_breakdown"]["by_visual_label"], {"severe_occlusion": 1})
        self.assertEqual(summary["guard_triggered_review_breakdown"]["by_navigation_assessment"], {"suspected_unsafe": 1})
        self.assertTrue(summary["acceptance_checks"]["target_unsafe_valid_reduced"])

    def test_overlay_displays_guard_risk_and_reason(self):
        records = [degraded_record(i) for i in range(8)] + [valid_record(8, min_support=6)]
        frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in records]
        record = apply_guard_to_records(records, OcclusionGuardConfig(), frames=frames)[-1]
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        rendered = draw_day68_overlay(frame, record)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(int(np.count_nonzero(rendered)), 0)
        self.assertFalse(np.array_equal(rendered, frame))

    def test_runner_writes_aligned_outputs_without_opening_frozen_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            day66 = root / "day66"
            day66.mkdir()
            episode = "development_ep"
            baseline = [degraded_record(i) for i in range(8)] + [
                valid_record(i, min_support=6) for i in range(8, 18)
            ]
            with (day66 / f"{episode}.jsonl").open("w", encoding="utf-8") as handle:
                for row in baseline:
                    handle.write(json.dumps(row) + "\n")

            video_path = root / "development.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
            )
            self.assertTrue(writer.isOpened())
            for index in range(len(baseline)):
                writer.write(np.full((180, 320, 3), index * 10, dtype=np.uint8))
            writer.release()

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {"episode": episode, "role": "temporal_development", "video_path": str(video_path)},
                            {"episode": "frozen_ep", "role": "frozen_same_source_holdout", "video_path": str(root / "must_not_open.mp4")},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            annotations_path = root / "annotations.jsonl"
            annotations_path.write_text(
                json.dumps(
                    {
                        "event_id": "development_ep__000008_000008__valid__stable_valid_output",
                        "episode": episode,
                        "primary_visual_label": "severe_occlusion",
                        "navigation_assessment": "suspected_unsafe",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            day67_results = root / "day67_results.json"
            day67_results.write_text(
                json.dumps(
                    {
                        "marker": "DAY67_FAILURE_TAXONOMY_COMPLETE",
                        "frozen_frames_accessed": 0,
                        "packet_metadata": {"frozen_frames_accessed": 0},
                    }
                ),
                encoding="utf-8",
            )
            day66_results = day66 / "day66_results.json"
            day66_results.write_text(
                json.dumps({"marker": "DAY66_OFFLINE_PILOT_COMPLETE"}) + "\n",
                encoding="utf-8",
            )
            day66_results_sha256 = hashlib.sha256(day66_results.read_bytes()).hexdigest()
            config_path = root / "config.json"
            config_payload = dict(OcclusionGuardConfig().__dict__)
            config_payload["upstream_day66_result_sha256"] = day66_results_sha256
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            output = root / "output"

            result = run_day68(
                day66_output=day66,
                manifest_path=manifest_path,
                annotations_path=annotations_path,
                day67_results_path=day67_results,
                config_path=config_path,
                output_dir=output,
                render_overlays=True,
            )

            self.assertEqual(result["marker"], "DAY68_SEVERE_OCCLUSION_COMPLETE")
            self.assertEqual(result["development_episode_count"], 1)
            self.assertEqual(result["frozen_video_frames_accessed"], 0)
            self.assertTrue(result["day68_engineering_gate_passed"])
            self.assertEqual(
                result["input_hashes"]["upstream_day66_result_sha256"],
                day66_results_sha256,
            )
            self.assertTrue(result["episode_results"][0]["overlay_decode_check"]["passed"])
            self.assertEqual(
                result["episode_results"][0]["overlay_decode_check"]["decoded_frame_count"],
                len(baseline),
            )
            self.assertTrue((output / "development_ep_day68.jsonl").is_file())
            self.assertTrue((output / "development_ep_day68_overlay.mp4").is_file())
            self.assertTrue((output / "day68_results.json").is_file())


if __name__ == "__main__":
    unittest.main()
