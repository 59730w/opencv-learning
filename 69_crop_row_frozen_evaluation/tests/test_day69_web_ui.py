from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))


class Day69WebUiTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("day69_web_ui")
        except ModuleNotFoundError:
            self.fail("Day69 web UI module is missing")

    def test_device_labels_map_to_cli_values(self):
        module = self._module()
        self.assertIsNone(module.resolve_device("自动选择"))
        self.assertEqual(module.resolve_device("CPU"), "cpu")
        self.assertEqual(module.resolve_device("CUDA"), "cuda")
        with self.assertRaisesRegex(ValueError, "设备"):
            module.resolve_device("TPU")

    def test_missing_or_unsupported_upload_is_rejected_before_inference(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "选择视频"):
                module.run_uploaded_video(None, "CPU", output_root=root)
            text_file = root / "not-video.txt"
            text_file.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "视频格式"):
                module.run_uploaded_video(text_file, "CPU", output_root=root)

    def test_uploaded_video_uses_isolated_output_and_returns_all_artifacts(self):
        module = self._module()
        calls = []

        def fake_runner(args):
            calls.append(args)
            episode_dir = Path(args.output_dir) / "field"
            episode_dir.mkdir(parents=True)
            paths = {
                "overlay_video": episode_dir / "field_overlay.mp4",
                "output_csv": episode_dir / "field_frames.csv",
                "output_jsonl": episode_dir / "field_frames.jsonl",
                "report_path": episode_dir / "field_report.json",
            }
            for path in paths.values():
                path.write_bytes(b"demo")
            aggregate_path = Path(args.output_dir) / "pilot_report.json"
            aggregate_path.write_text("{}", encoding="utf-8")
            return {
                "marker": "DAY69_OFFLINE_PILOT_COMPLETE",
                "frame_count": 12,
                "navigation_invariant_violations": 0,
                "reports": [{
                    **{key: str(path) for key, path in paths.items()},
                    "status_counts": {
                        "valid": 3,
                        "candidate": 1,
                        "degraded": 7,
                        "reject": 1,
                    },
                    "overlay_verification": {"passed": True},
                }],
                "report_path": str(aggregate_path),
                "claim_boundary": "Offline diagnostic pilot only",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "田间 sample.mp4"
            video.write_bytes(b"video")
            result = module.run_uploaded_video(
                video,
                "CUDA",
                output_root=root / "runs",
                runner=fake_runner,
                run_id="demo-run",
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].device, "cuda")
            self.assertEqual(Path(calls[0].output_dir).name, "demo-run-field-sample")
            self.assertEqual(result.overlay_video.name, "field_overlay.mp4")
            self.assertEqual(len(result.downloads), 5)
            self.assertIn("3", result.status_markdown)
            self.assertIn("仅 `valid`", result.status_markdown)
            self.assertEqual(result.summary["frame_count"], 12)
            self.assertEqual(result.summary["navigation_invariant_violations"], 0)

    def test_duplicate_run_directory_is_not_overwritten(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "field.mp4"
            video.write_bytes(b"video")
            occupied = root / "runs" / "same-field"
            occupied.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "已存在"):
                module.run_uploaded_video(
                    video,
                    "CPU",
                    output_root=root / "runs",
                    runner=lambda args: {},
                    run_id="same",
                )

    def test_public_interface_uses_product_identity_without_course_day_text(self):
        module = self._module()
        config = module.build_app().get_config_file()
        visible_strings = [str(config.get("title", ""))]
        for component in config.get("components", []):
            props = component.get("props", {})
            for key in ("label", "info", "value"):
                if isinstance(props.get(key), str):
                    visible_strings.append(props[key])
        visible_text = "\n".join(visible_strings)
        self.assertIn("禾迹", visible_text)
        self.assertNotIn("Day69", visible_text)
        self.assertNotIn("DAY 69", visible_text)


if __name__ == "__main__":
    unittest.main()
