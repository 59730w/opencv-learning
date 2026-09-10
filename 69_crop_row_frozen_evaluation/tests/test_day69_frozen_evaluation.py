from __future__ import annotations

import csv
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

from day69_frozen_evaluation import (  # noqa: E402
    FROZEN_VIDEO_ROLE,
    evaluate_static_entries,
    geometry_summary,
    rowdetr_reference_rows,
    run_single_video,
    select_frozen_video_entries,
    verify_frozen_protocol,
)
from run_crop_row_pilot import discover_video_inputs  # noqa: E402

DAY63_CODE = Path(__file__).resolve().parents[2] / "63_crop_row_geometry_extraction" / "code"
sys.path.insert(0, str(DAY63_CODE))
from day63_crop_row_geometry import CropRowLine  # noqa: E402

DAY65_CODE = Path(__file__).resolve().parents[2] / "65_crop_row_video_temporal_stability" / "code"
sys.path.insert(0, str(DAY65_CODE))
from day65_video_temporal import TemporalConfig  # noqa: E402

DAY68_CODE = Path(__file__).resolve().parents[2] / "68_crop_row_severe_occlusion" / "code"
sys.path.insert(0, str(DAY68_CODE))
from day68_severe_occlusion import NAVIGATION_FIELDS, OcclusionGuardConfig  # noqa: E402


class ConstantPredictor:
    def predict(self, frames: list[np.ndarray]) -> list[tuple[CropRowLine, ...]]:
        rows = (
            CropRowLine(far_x_norm=0.42, near_x_norm=0.24, confidence=0.9, support_band_count=9),
            CropRowLine(far_x_norm=0.58, near_x_norm=0.76, confidence=0.9, support_band_count=9),
        )
        return [rows for _ in frames]


class Day69FrozenEvaluationTests(unittest.TestCase):
    def test_input_discovery_is_recursive_sorted_and_video_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "b.MP4").write_bytes(b"")
            (root / "nested" / "a.avi").write_bytes(b"")
            (root / "ignore.txt").write_text("x", encoding="utf-8")
            videos = discover_video_inputs(root)
            self.assertEqual([item.name for item in videos], ["b.MP4", "a.avi"])
            self.assertEqual(discover_video_inputs(root / "b.MP4"), [(root / "b.MP4").resolve()])
    def test_protocol_verification_detects_artifact_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"frozen")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            protocol = {
                "marker": "DAY69_PREACCESS_FROZEN",
                "frozen_artifacts": {"artifact": {"path": str(artifact), "sha256": digest}},
            }
            evidence = verify_frozen_protocol(protocol)
            self.assertEqual(evidence["artifact"]["sha256"], digest)
            artifact.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_frozen_protocol(protocol)

    def test_only_exact_frozen_video_role_is_selected(self):
        manifest = {
            "episodes": [
                {"episode": "dev", "role": "temporal_development", "video_path": "dev.mp4"},
                {"episode": "f0", "role": FROZEN_VIDEO_ROLE, "video_path": "f0.mp4"},
                {"episode": "f1", "role": FROZEN_VIDEO_ROLE, "video_path": "f1.mp4"},
            ]
        }
        selected = select_frozen_video_entries(manifest, expected_count=2)
        self.assertEqual([row["episode"] for row in selected], ["f0", "f1"])
        with self.assertRaisesRegex(ValueError, "expected 3"):
            select_frozen_video_entries(manifest, expected_count=3)

    def test_rowdetr_polylines_become_ordered_reference_rows(self):
        payload = {
            "labels": [
                {"x": [40, 36, 32, 28], "y": [90, 70, 50, 40]},
                {"x": [60, 64, 68, 72], "y": [90, 70, 50, 40]},
            ]
        }
        rows, rejected = rowdetr_reference_rows(payload, width=101, height=101)
        self.assertEqual(rejected, 0)
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0].far_x_norm, rows[1].far_x_norm)
        self.assertAlmostEqual(rows[0].near_x_norm, 0.40, places=2)
        self.assertAlmostEqual(rows[1].near_x_norm, 0.60, places=2)

    def test_rowdetr_short_polylines_are_explicitly_not_evaluable(self):
        payload = {"labels": [{"x": [30, 31, 32, 33], "y": [10, 20, 25, 30]}]}
        rows, rejected = rowdetr_reference_rows(payload, width=101, height=101)
        self.assertEqual(rows, ())
        self.assertEqual(rejected, 1)

    def test_geometry_summary_keeps_positive_only_safety_unavailable(self):
        records = [
            {
                "predicted_count": 2,
                "reference_count": 2,
                "matched_count": 2,
                "position_error_sum": 0.02,
                "heading_error_sum": 2.0,
                "reference_corridor_available": True,
                "prediction_corridor_available": True,
                "boundary_pair_correct": True,
                "corridor_center_error_norm": 0.01,
                "prediction_status": "valid",
                "runtime_ms": 20.0,
            }
        ]
        summary = geometry_summary(records, evidence_role="frozen_external_test_positive_only")
        self.assertEqual(summary["row_detection_precision"], 1.0)
        self.assertEqual(summary["row_detection_recall"], 1.0)
        self.assertIn("unsafe_false_valid_rate", summary["unavailable_metrics"])
        self.assertNotIn("unsafe_false_valid_rate", summary["gates"])

    def test_geometry_summary_applies_frozen_absolute_gates(self):
        records = [
            {
                "predicted_count": 2,
                "reference_count": 2,
                "matched_count": 2,
                "position_error_sum": 0.04,
                "heading_error_sum": 6.0,
                "reference_corridor_available": True,
                "prediction_corridor_available": True,
                "boundary_pair_correct": True,
                "corridor_center_error_norm": 0.03,
                "prediction_status": "valid",
                "runtime_ms": 35.0,
            }
        ]
        summary = geometry_summary(records, evidence_role="same_source_internal_benchmark")
        self.assertTrue(summary["all_supported_absolute_gates_passed"])
        self.assertAlmostEqual(summary["matched_bottom_position_mae_norm"], 0.02)
        self.assertAlmostEqual(summary["matched_heading_mae_deg"], 3.0)

    def test_single_video_emits_aligned_jsonl_csv_overlay_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90)
            )
            self.assertTrue(writer.isOpened())
            for index in range(12):
                writer.write(np.full((90, 160, 3), index * 10, dtype=np.uint8))
            writer.release()

            result = run_single_video(
                video,
                output_dir=root / "output",
                predictor=ConstantPredictor(),
                temporal_config=TemporalConfig(),
                occlusion_config=OcclusionGuardConfig(),
                use_optical_flow=False,
                episode="synthetic",
                evidence_role="user_input_unverified",
            )

            self.assertEqual(result["source_frame_count"], 12)
            self.assertTrue(result["overlay_verification"]["passed"])
            self.assertEqual(result["overlay_verification"]["width"], 320)
            self.assertEqual(result["navigation_invariant_violations"], 0)
            for key in ("output_jsonl", "output_csv", "overlay_video", "report_path"):
                self.assertTrue(Path(result[key]).is_file(), key)
            jsonl_rows = [
                json.loads(line)
                for line in Path(result["output_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]
            with Path(result["output_csv"]).open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(jsonl_rows), 12)
            self.assertEqual(len(csv_rows), 12)
            for row in jsonl_rows:
                if row["pilot_state"] != "valid":
                    self.assertFalse(row["navigation_available"])
                    for field in NAVIGATION_FIELDS:
                        self.assertIsNone(row[field])

    def test_static_evaluation_verifies_hashes_and_writes_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.jpg"
            label_path = root / "sample.json"
            cv2.imwrite(str(image_path), np.zeros((101, 101, 3), dtype=np.uint8))
            label_path.write_text(
                json.dumps({"labels": [
                    {"x": [42, 42], "y": [40, 90]},
                    {"x": [58, 58], "y": [40, 90]},
                ]}),
                encoding="utf-8",
            )
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            entries = [{
                "item_id": "sample", "image_path": "sample.jpg",
                "label_path": "sample.json", "image_sha256": digest,
            }]
            output = root / "records.jsonl"
            result = evaluate_static_entries(
                entries, root=root, label_kind="rowdetr_json",
                predictor=ConstantPredictor(), evidence_role="frozen_external_test_positive_only",
                output_jsonl=output,
            )
            self.assertEqual(result["summary"]["record_count"], 1)
            self.assertEqual(result["image_hashes_verified"], 1)
            self.assertTrue(output.is_file())
            image_path.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate_static_entries(
                    entries, root=root, label_kind="rowdetr_json",
                    predictor=ConstantPredictor(), evidence_role="frozen_external_test_positive_only",
                    output_jsonl=output,
                )


if __name__ == "__main__":
    unittest.main()
