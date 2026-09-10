from __future__ import annotations

import importlib
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
DAY63_CODE = Path(__file__).resolve().parents[2] / "63_crop_row_geometry_extraction" / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(DAY63_CODE))

from day63_crop_row_geometry import CropRowLine  # noqa: E402


class Day69V2EvaluationTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("day69_v2_evaluation")
        except ModuleNotFoundError:
            self.fail("Day69 v2 evaluator module is missing")

    def test_partial_polyline_is_evaluated_on_visible_overlap(self):
        module = self._module()
        payload = {
            "labels": [
                {"x": [42, 38, 34, 30], "y": [55, 70, 85, 95]},
            ]
        }
        references, rejected = module.visible_reference_polylines(
            payload, width=101, height=101, min_overlap_y_norm=0.20
        )
        self.assertEqual(rejected, 0)
        self.assertEqual(len(references), 1)
        prediction = CropRowLine(
            far_x_norm=0.46,
            near_x_norm=0.32,
            confidence=0.9,
            support_band_count=9,
        )
        result = module.match_visible_polylines(
            (prediction,), references, max_mean_x_error_norm=0.03
        )
        self.assertEqual(result["matched_count"], 1)
        self.assertLess(result["mean_x_error_norm"], 0.02)
        self.assertGreaterEqual(result["evaluated_overlap_y_norm"], 0.20)

    def test_short_visible_span_is_explicitly_rejected(self):
        module = self._module()
        payload = {"labels": [{"x": [30, 31, 32], "y": [70, 75, 80]}]}
        references, rejected = module.visible_reference_polylines(
            payload, width=101, height=101, min_overlap_y_norm=0.20
        )
        self.assertEqual(references, ())
        self.assertEqual(rejected, 1)

    def test_matching_maximizes_pair_count_before_minimizing_error(self):
        module = self._module()
        references = (
            module.VisiblePolyline((0.40, 0.40), (0.50, 0.90)),
            module.VisiblePolyline((0.50, 0.50), (0.50, 0.90)),
        )
        predictions = (
            CropRowLine(0.49, 0.49, 0.9, 9),
            CropRowLine(0.58, 0.58, 0.9, 9),
        )
        result = module.match_visible_polylines(
            predictions, references, max_mean_x_error_norm=0.10
        )
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["pairs"], [[0, 0], [1, 1]])

    def test_archive_group_split_is_deterministic_and_group_disjoint(self):
        module = self._module()
        names = [
            f"{split}/{group}_mp4-{frame:04d}_jpg.rf.{frame:032x}.jpg"
            for split in ("train", "valid", "test")
            for group in ("video_a", "video_b", "video_c", "video_d")
            for frame in range(3)
        ]
        frozen = module.freeze_archive_groups(
            names,
            seed="unit-test",
            minimum_group_images=3,
            reserved_group_fraction=0.25,
            maximum_images_per_group=2,
        )
        repeated = module.freeze_archive_groups(
            list(reversed(names)),
            seed="unit-test",
            minimum_group_images=3,
            reserved_group_fraction=0.25,
            maximum_images_per_group=2,
        )
        self.assertEqual(frozen, repeated)
        self.assertEqual(len(frozen["reserved_groups"]), 1)
        self.assertEqual(len(frozen["reserved_entries"]), 2)
        reserved = set(frozen["reserved_groups"])
        self.assertFalse(reserved.intersection(frozen["development_groups"]))

    def test_yolo_polygon_becomes_visible_centerline_without_rasterization(self):
        module = self._module()
        label = "0 0.30 0.40 0.35 0.90 0.45 0.90 0.40 0.40\n"
        references, rejected = module.yolo_segmentation_references(
            label, min_overlap_y_norm=0.20
        )
        self.assertEqual(rejected, 0)
        self.assertEqual(len(references), 1)
        self.assertAlmostEqual(references[0].x_at(module.np.asarray([0.40]))[0], 0.35, places=3)
        self.assertAlmostEqual(references[0].x_at(module.np.asarray([0.90]))[0], 0.40, places=3)
        prediction = CropRowLine(0.35, 0.40, 0.9, 9)
        matching = module.match_visible_polylines(
            (prediction,), references, max_mean_x_error_norm=0.06
        )
        self.assertEqual(matching["matched_count"], 1)

    def test_central_row_summary_marks_unlabelled_other_rows_unavailable(self):
        module = self._module()
        summary = module.central_row_summary(
            [
                {
                    "reference_available": True,
                    "matched": True,
                    "visible_position_error_norm": 0.01,
                    "heading_error_deg": 2.0,
                    "runtime_ms": 25.0,
                }
            ],
            evidence_role="independent_source_later_session_central_row_positive_only",
        )
        self.assertTrue(summary["evaluation_valid"])
        self.assertEqual(summary["central_row_detection_recall"], 1.0)
        self.assertIsNone(summary["all_row_detection_precision"])
        self.assertIn("all_row_detection_precision", summary["unavailable_metrics"])
        self.assertNotIn("all_row_detection_precision", summary["gates"])
        self.assertNotIn("runtime_median_ms_per_frame", summary["gates"])
        self.assertEqual(
            summary["runtime_gate_status"], "UNAVAILABLE_MIXED_RESOLUTION_AND_WARMUP"
        )

    def test_central_row_summary_fails_closed_without_reference_support(self):
        module = self._module()
        summary = module.central_row_summary(
            [{"reference_available": False, "matched": False, "runtime_ms": 25.0}],
            evidence_role="external",
        )
        self.assertFalse(summary["evaluation_valid"])
        self.assertEqual(summary["evaluation_status"], "INVALID_REFERENCE_SUPPORT")
        self.assertIsNone(summary["central_row_detection_recall"])

    def test_ssr_evaluator_verifies_hashes_and_emits_supported_metrics(self):
        module = self._module()

        class Predictor:
            last_runtime_ms_per_frame = 18.0

            def predict(self, frames):
                return [(CropRowLine(0.35, 0.40, 0.9, 9),) for _ in frames]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.jpg"
            label_path = root / "sample.txt"
            cv2.imwrite(str(image_path), np.full((100, 100, 3), 120, dtype=np.uint8))
            label_path.write_text(
                "0 0.30 0.40 0.35 0.90 0.45 0.90 0.40 0.40\n",
                encoding="utf-8",
            )
            entry = {
                "item_id": "sample",
                "image_path": image_path.name,
                "label_path": label_path.name,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            }
            output = root / "records.jsonl"
            result = module.evaluate_ssr_entries(
                [entry],
                root=root,
                predictor=Predictor(),
                output_jsonl=output,
                evidence_role="external_central_row",
                batch_size=1,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["hash_verified_pair_count"], 1)
            self.assertTrue(result["summary"]["evaluation_valid"])
            self.assertEqual(result["summary"]["central_row_detection_recall"], 1.0)
            self.assertTrue(record["matched"])
            self.assertIsNone(result["summary"]["all_row_detection_precision"])

    def test_zero_reference_denominator_fails_closed(self):
        module = self._module()
        summary = module.geometry_summary_v2(
            [
                {
                    "predicted_count": 2,
                    "reference_count": 0,
                    "matched_count": 0,
                    "runtime_ms": 20.0,
                }
            ],
            evidence_role="external_positive_geometry",
        )
        self.assertFalse(summary["evaluation_valid"])
        self.assertEqual(summary["evaluation_status"], "INVALID_REFERENCE_SUPPORT")
        self.assertIsNone(summary["row_detection_precision"])
        self.assertIsNone(summary["row_detection_recall"])
        self.assertNotIn("row_detection_precision", summary["gates"])
        self.assertFalse(summary["all_supported_absolute_gates_passed"])

    def test_nonzero_reference_with_no_predictions_is_a_real_zero(self):
        module = self._module()
        summary = module.geometry_summary_v2(
            [
                {
                    "predicted_count": 0,
                    "reference_count": 2,
                    "matched_count": 0,
                    "runtime_ms": 20.0,
                }
            ],
            evidence_role="external_positive_geometry",
        )
        self.assertTrue(summary["evaluation_valid"])
        self.assertEqual(summary["row_detection_precision"], 0.0)
        self.assertEqual(summary["row_detection_recall"], 0.0)
        self.assertFalse(summary["gates"]["row_detection_recall"]["passed"])


if __name__ == "__main__":
    unittest.main()
