import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "code" / "day71_device_parity.py"
)
SPEC = importlib.util.spec_from_file_location("day71_device_parity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

WRAPPER_PATH = MODULE_PATH.with_name("run_crop_row_pilot_device_stable.py")
WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "run_crop_row_pilot_device_stable", WRAPPER_PATH
)
WRAPPER = importlib.util.module_from_spec(WRAPPER_SPEC)
sys.modules[WRAPPER_SPEC.name] = WRAPPER
WRAPPER_SPEC.loader.exec_module(WRAPPER)


class ArrayEvidenceTests(unittest.TestCase):
    def test_array_fingerprint_changes_with_shape_and_dtype(self):
        base = np.arange(6, dtype=np.uint8)
        reshaped = base.reshape(2, 3)
        wider = base.astype(np.uint16)

        self.assertNotEqual(MODULE.array_fingerprint(base), MODULE.array_fingerprint(reshaped))
        self.assertNotEqual(MODULE.array_fingerprint(base), MODULE.array_fingerprint(wider))

    def test_probability_comparison_reports_threshold_crossings_by_frame(self):
        left = np.array([[[0.19, 0.20], [0.21, 0.80]], [[0.10, 0.30], [0.40, 0.50]]])
        right = np.array([[[0.21, 0.20], [0.19, 0.80]], [[0.10, 0.31], [0.40, 0.50]]])

        result = MODULE.compare_probability_stacks(left, right, threshold=0.20)

        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["frames_with_threshold_crossings"], 1)
        self.assertEqual(result["threshold_crossing_pixels"], 2)
        self.assertAlmostEqual(result["max_abs_difference"], 0.02)


class RecordComparisonTests(unittest.TestCase):
    def test_record_comparison_finds_first_mismatch_and_runs(self):
        left = [
            {"frame_index": 0, "pilot_state": "degraded", "navigation_available": False},
            {"frame_index": 1, "pilot_state": "valid", "navigation_available": True},
            {"frame_index": 2, "pilot_state": "valid", "navigation_available": True},
            {"frame_index": 3, "pilot_state": "degraded", "navigation_available": False},
        ]
        right = [
            {"frame_index": 0, "pilot_state": "degraded", "navigation_available": False},
            {"frame_index": 1, "pilot_state": "degraded", "navigation_available": False},
            {"frame_index": 2, "pilot_state": "degraded", "navigation_available": False},
            {"frame_index": 3, "pilot_state": "degraded", "navigation_available": False},
        ]

        result = MODULE.compare_final_records(left, right)

        self.assertEqual(result["status_mismatch_frames"], 2)
        self.assertEqual(result["navigation_mismatch_frames"], 2)
        self.assertEqual(result["first_status_mismatch_frame"], 1)
        self.assertEqual(result["status_mismatch_runs"], [[1, 2]])

    def test_root_cause_classification_requires_isolated_evidence(self):
        comparisons = {
            "cpu_eager_bs1__cpu_optimized_bs1": {"status_mismatch_frames": 4},
            "cpu_eager_bs1__cuda_eager_bs1": {"status_mismatch_frames": 0},
            "cuda_eager_bs1__cuda_eager_bs32": {"status_mismatch_frames": 0},
        }

        result = MODULE.classify_root_cause(comparisons)

        self.assertEqual(result["status"], "ROOT_CAUSE_IDENTIFIED")
        self.assertEqual(result["primary_factor"], "cpu_optimized_backend")
        self.assertIn("isolated", result["reason"])

    def test_policy_prefers_matched_eager_batch_when_it_removes_state_difference(self):
        comparisons = {
            "cpu_eager_bs1__cpu_optimized_bs1": {"status_mismatch_frames": 4},
            "cpu_eager_bs1__cuda_eager_bs1": {"status_mismatch_frames": 0},
            "cuda_eager_bs1__cuda_eager_bs32": {"status_mismatch_frames": 3},
            "cpu_optimized_bs1__cuda_eager_bs32": {"status_mismatch_frames": 7},
        }

        policy = MODULE.recommend_execution_policy(comparisons)

        self.assertEqual(policy["status"], "PASS_DEVELOPMENT_REPRODUCIBILITY")
        self.assertEqual(policy["policy"], "matched_eager_batch1")
        self.assertFalse(policy["changes_model_or_thresholds"])

    def test_policy_prefers_no_tf32_cuda_when_measured_parity_passes(self):
        comparisons = {
            "cpu_eager_bs1__cpu_optimized_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_eager_bs1": {"status_mismatch_frames": 14},
            "cuda_eager_bs1__cuda_eager_bs32": {"status_mismatch_frames": 31},
            "cpu_optimized_bs1__cuda_eager_bs32": {"status_mismatch_frames": 37},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs1": {"status_mismatch_frames": 0},
        }

        policy = MODULE.recommend_execution_policy(comparisons)

        self.assertEqual(policy["status"], "PASS_DEVELOPMENT_REPRODUCIBILITY")
        self.assertEqual(policy["policy"], "cuda_no_tf32_eager_batch1")

    def test_policy_prefers_faster_no_tf32_batch32_when_it_also_passes(self):
        comparisons = {
            "cpu_eager_bs1__cpu_optimized_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_eager_bs1": {"status_mismatch_frames": 14},
            "cuda_eager_bs1__cuda_eager_bs32": {"status_mismatch_frames": 31},
            "cpu_optimized_bs1__cuda_eager_bs32": {"status_mismatch_frames": 37},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 0},
            "cpu_optimized_bs1__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 0},
        }

        policy = MODULE.recommend_execution_policy(comparisons)

        self.assertEqual(policy["policy"], "cuda_no_tf32_eager_batch32")

    def test_policy_keeps_fixed_device_when_production_cpu_still_differs(self):
        comparisons = {
            "cpu_eager_bs1__cpu_optimized_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_eager_bs1": {"status_mismatch_frames": 14},
            "cuda_eager_bs1__cuda_eager_bs32": {"status_mismatch_frames": 31},
            "cpu_optimized_bs1__cuda_eager_bs32": {"status_mismatch_frames": 37},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 0},
            "cpu_optimized_bs1__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 1},
        }

        policy = MODULE.recommend_execution_policy(comparisons)

        self.assertEqual(policy["policy"], "explicit_cpu_reference")

    def test_root_cause_identifies_tf32_when_both_no_tf32_batches_match_cpu(self):
        comparisons = {
            "cpu_eager_bs1__cpu_optimized_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_eager_bs1": {"status_mismatch_frames": 14},
            "cuda_eager_bs1__cuda_eager_bs32": {"status_mismatch_frames": 31},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 0},
            "cuda_eager_bs32__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 37},
        }

        result = MODULE.classify_root_cause(comparisons)

        self.assertEqual(result["status"], "ROOT_CAUSE_IDENTIFIED")
        self.assertEqual(result["primary_factor"], "cudnn_tf32_execution_path")

    def test_root_cause_does_not_overclaim_when_batch32_tf32_contrast_is_zero(self):
        comparisons = {
            "cpu_eager_bs1__cpu_optimized_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_eager_bs1": {"status_mismatch_frames": 14},
            "cuda_eager_bs1__cuda_eager_bs32": {"status_mismatch_frames": 2},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs1": {"status_mismatch_frames": 0},
            "cpu_eager_bs1__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 0},
            "cuda_eager_bs32__cuda_no_tf32_eager_bs32": {"status_mismatch_frames": 0},
        }

        result = MODULE.classify_root_cause(comparisons)

        self.assertEqual(result["status"], "PARTIALLY_IDENTIFIED")


class ModelLayerTests(unittest.TestCase):
    def test_model_runner_uses_declared_batches_and_returns_probabilities(self):
        model = torch.nn.Conv2d(4, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.25)
        features = np.arange(3 * 4 * 2 * 2, dtype=np.uint8).reshape(3, 4, 2, 2)

        first = MODULE.run_model_probabilities(
            model, features, device="cpu", batch_size=2, channels_last=False
        )
        second = MODULE.run_model_probabilities(
            model, features, device="cpu", batch_size=2, channels_last=False
        )

        self.assertEqual(first.shape, (3, 2, 2))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.logical_and(first > 0.0, first < 1.0).all())

    def test_observation_comparison_separates_count_and_geometry(self):
        left = [
            [
                {"near_x_norm": 0.2, "far_x_norm": 0.4, "confidence": 0.8},
                {"near_x_norm": 0.8, "far_x_norm": 0.6, "confidence": 0.7},
            ],
            [],
        ]
        right = [
            [
                {"near_x_norm": 0.21, "far_x_norm": 0.4, "confidence": 0.79},
                {"near_x_norm": 0.8, "far_x_norm": 0.6, "confidence": 0.7},
            ],
            [{"near_x_norm": 0.5, "far_x_norm": 0.5, "confidence": 0.3}],
        ]

        result = MODULE.compare_observations(left, right)

        self.assertEqual(result["row_count_mismatch_frames"], 1)
        self.assertEqual(result["first_row_count_mismatch_frame"], 1)
        self.assertAlmostEqual(result["max_matched_near_x_abs_difference"], 0.01)


class CommandContractTests(unittest.TestCase):
    def test_stable_runner_enforces_validated_cuda_batch32(self):
        self.assertEqual(
            WRAPPER.with_validated_execution(["--input", "field.mp4"]),
            ["--input", "field.mp4", "--device", "cuda", "--batch-size", "32"],
        )
        with self.assertRaisesRegex(ValueError, "batch-size 32"):
            WRAPPER.with_validated_execution(
                ["--input", "field.mp4", "--batch-size", "1"]
            )
        with self.assertRaisesRegex(ValueError, "device cuda"):
            WRAPPER.with_validated_execution(
                ["--input", "field.mp4", "--device", "cpu"]
            )

    def test_stable_runner_disables_cudnn_tf32_without_changing_matmul_policy(self):
        original_cudnn = torch.backends.cudnn.allow_tf32
        original_matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = True
            metadata = WRAPPER.configure_device_stable_execution()

            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(
                torch.backends.cuda.matmul.allow_tf32, original_matmul
            )
            self.assertEqual(metadata["policy"], "cuda_no_tf32_eager_batch32")
            self.assertFalse(metadata["changes_model_or_thresholds"])
        finally:
            torch.backends.cudnn.allow_tf32 = original_cudnn

    def test_stable_runner_persists_effective_policy_in_pilot_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "pilot_report.json"
            report.write_text('{"marker":"DAY69_OFFLINE_PILOT_COMPLETE"}', encoding="utf-8")
            policy = {"policy": "cuda_no_tf32_eager_batch32", "effective_batch_size": 32}

            WRAPPER.persist_policy(report, policy)

            payload = __import__("json").loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["day71_execution_policy"], policy)

    def test_emitted_jsonl_comparison_includes_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "cpu.jsonl"
            right = root / "cuda.jsonl"
            left.write_text(
                '{"frame_index":0,"pilot_state":"valid","navigation_available":true,"lateral_offset_norm":0.1}\n',
                encoding="utf-8",
            )
            right.write_text(
                '{"frame_index":0,"pilot_state":"valid","navigation_available":true,"lateral_offset_norm":0.2}\n',
                encoding="utf-8",
            )

            result = MODULE.compare_emitted_jsonl(left, right)

            self.assertEqual(result["status_mismatch_frames"], 0)
            self.assertEqual(result["geometry_mismatch_frames"], 1)

    def test_emitted_jsonl_comparison_accepts_only_bounded_float_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "cpu.jsonl"
            right = root / "cuda.jsonl"
            left.write_text(
                '{"frame_index":0,"pilot_state":"valid","navigation_available":true,'
                '"lateral_offset_norm":0.1,"crop_rows":[{"track_id":2,'
                '"near_x_norm":0.3,"status":"tracked"}]}\n',
                encoding="utf-8",
            )
            right.write_text(
                '{"frame_index":0,"pilot_state":"valid","navigation_available":true,'
                '"lateral_offset_norm":0.100005,"crop_rows":[{"track_id":2,'
                '"near_x_norm":0.300005,"status":"tracked"}]}\n',
                encoding="utf-8",
            )

            result = MODULE.compare_emitted_jsonl(
                left, right, geometry_abs_tolerance=1e-5
            )

            self.assertEqual(result["geometry_mismatch_frames"], 0)
            self.assertEqual(result["geometry_structural_mismatch_frames"], 0)
            self.assertAlmostEqual(result["geometry_max_abs_difference"], 5e-6)
            self.assertEqual(result["geometry_abs_tolerance"], 1e-5)

    def test_emitted_jsonl_comparison_rejects_frame_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "cpu.jsonl"
            right = root / "cuda.jsonl"
            left.write_text(
                '{"episode":"ep","frame_index":0,"pilot_state":"degraded",'
                '"navigation_available":false}\n',
                encoding="utf-8",
            )
            right.write_text(
                '{"episode":"ep","frame_index":1,"pilot_state":"degraded",'
                '"navigation_available":false}\n',
                encoding="utf-8",
            )

            result = MODULE.compare_emitted_jsonl(left, right)

            self.assertEqual(result["frame_identity_mismatch_frames"], 1)

    def test_cli_provenance_binds_jsonl_hash_path_video_and_policy(self):
        import hashlib
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            episode = "ep"
            paths = {}
            for role in ("cpu", "stable"):
                jsonl = root / f"{role}.jsonl"
                jsonl.write_text(
                    json.dumps(
                        {
                            "episode": episode,
                            "frame_index": 0,
                            "pilot_state": "degraded",
                            "navigation_available": False,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                report = {
                    "marker": "DAY69_OFFLINE_PILOT_COMPLETE",
                    "input": str(video),
                    "frame_count": 1,
                    "device": "cpu" if role == "cpu" else "cuda",
                    "predictor_backend": (
                        "torchscript_optimize_for_inference_channels_last"
                        if role == "cpu"
                        else "torch_eager"
                    ),
                    "reports": [
                        {
                            "episode": episode,
                            "source_video": str(video),
                            "output_jsonl": str(jsonl),
                            "artifacts_sha256": {
                                "jsonl": hashlib.sha256(jsonl.read_bytes()).hexdigest()
                            },
                        }
                    ],
                }
                if role == "stable":
                    report["day71_execution_policy"] = {
                        "policy": "cuda_no_tf32_eager_batch32",
                        "effective_device": "cuda",
                        "effective_batch_size": 32,
                        "cudnn_allow_tf32": False,
                    }
                report_path = root / f"{role}_report.json"
                report_path.write_text(json.dumps(report), encoding="utf-8")
                paths[role] = (jsonl, report_path)

            result = MODULE.validate_cli_provenance(
                video_path=video,
                cpu_jsonl=paths["cpu"][0],
                cpu_report=paths["cpu"][1],
                stable_jsonl=paths["stable"][0],
                stable_report=paths["stable"][1],
            )

            self.assertTrue(result["valid"])
            self.assertTrue(all(result["checks"].values()))

            cpu_text = paths["cpu"][0].read_text(encoding="utf-8")
            paths["cpu"][0].write_text(
                cpu_text.replace('"degraded"', '"candidate"'), encoding="utf-8"
            )
            tampered = MODULE.validate_cli_provenance(
                video_path=video,
                cpu_jsonl=paths["cpu"][0],
                cpu_report=paths["cpu"][1],
                stable_jsonl=paths["stable"][0],
                stable_report=paths["stable"][1],
            )
            self.assertFalse(tampered["valid"])
            self.assertFalse(tampered["checks"]["reports_bind_compared_jsonl_hashes"])

    def test_parser_exposes_six_arm_defaults(self):
        parser = MODULE.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.video.name, "crowfollow_lecropfollow_exp_2026-01-26-12-35-02_ep0.mp4")
        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.peak_height, 0.20)

    def test_diagnosis_rejects_missing_required_input_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = MODULE.build_parser().parse_args(
                [
                    "--video", str(root / "missing.mp4"),
                    "--checkpoint", str(root / "missing.pt"),
                    "--output-dir", str(root / "output"),
                    "--report", str(root / "report.json"),
                    "--differences-csv", str(root / "diff.csv"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "missing video"):
                MODULE.run_diagnosis(args)


if __name__ == "__main__":
    unittest.main()
