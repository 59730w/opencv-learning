from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


DAY70_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = DAY70_ROOT / "code"
REPO_ROOT = DAY70_ROOT.parent
sys.path.insert(0, str(CODE_DIR))


class Day70DeliveryTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("day70_delivery")
        except ModuleNotFoundError:
            self.fail("Day70 delivery checker module is missing")

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _minimal_manifest() -> dict:
        return {
            "schema_version": 1,
            "marker": "DAY70_DELIVERY_MANIFEST",
            "date": "2026-09-11",
            "baseline_commit": "43f0951c36ee45d9c4b5991e6305bca526451696",
            "frozen_artifacts": [],
            "evidence_checks": [],
            "claims": [
                {"id": "offline_pilot_delivery", "status": "PASS", "statement": "可运行离线Pilot"},
                {"id": "ssr_central_row_recall", "status": "FAILED", "statement": "Recall未达门槛"},
                {"id": "cross_device_exact_parity", "status": "FAILED", "statement": "CPU/CUDA逐帧状态不完全一致"},
                {"id": "all_row_external_generalization", "status": "NOT_ESTABLISHED", "statement": "没有完整多行外部真值"},
                {"id": "reject_aware_external_generalization", "status": "BLOCKED", "statement": "缺少目标域负样本"},
                {"id": "real_video_safety", "status": "BLOCKED", "statement": "缺少逐帧走廊有效性真值"},
                {"id": "metric_robot_measurement", "status": "BLOCKED", "statement": "缺少相机与车体标定"},
            ],
            "runtime_contract": {},
            "required_artifacts": [],
            "tracked_hygiene": {"enabled": False, "forbidden_patterns": []},
        }

    def test_sha256_file_matches_standard_library(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.bin"
            path.write_bytes(b"frozen evidence")
            self.assertEqual(module.sha256_file(path), hashlib.sha256(b"frozen evidence").hexdigest())

    def test_hash_drift_fails_closed(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "frozen.json"
            artifact.write_text("before", encoding="utf-8")
            manifest = self._minimal_manifest()
            manifest["frozen_artifacts"] = [{
                "path": "frozen.json",
                "sha256": hashlib.sha256(b"before").hexdigest(),
                "role": "frozen_result",
            }]
            manifest_path = root / "manifest.json"
            self._write_json(manifest_path, manifest)
            artifact.write_text("after", encoding="utf-8")

            report = module.verify_delivery(root, manifest_path)

            self.assertFalse(report["passed"])
            self.assertTrue(any(item["id"] == "frozen_hash:frozen.json" for item in report["failures"]))

    def test_evidence_value_mismatch_is_reported(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_json(root / "source.json", {"metrics": {"recall": 0.5}})
            manifest = self._minimal_manifest()
            manifest["evidence_checks"] = [{
                "id": "external_recall",
                "source": "source.json",
                "json_path": "metrics.recall",
                "expected": 0.8,
            }]
            manifest_path = root / "manifest.json"
            self._write_json(manifest_path, manifest)

            report = module.verify_delivery(root, manifest_path)

            self.assertFalse(report["passed"])
            failure = next(item for item in report["failures"] if item["id"] == "evidence:external_recall")
            self.assertEqual(failure["actual"], 0.5)

    def test_failed_external_recall_cannot_be_relabelled_pass(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._minimal_manifest()
            manifest["claims"][1]["status"] = "PASS"
            manifest_path = root / "manifest.json"
            self._write_json(manifest_path, manifest)

            report = module.verify_delivery(root, manifest_path)

            self.assertFalse(report["passed"])
            self.assertTrue(any(item["id"] == "claim:ssr_central_row_recall" for item in report["failures"]))

    def test_cross_device_exact_parity_claim_cannot_be_omitted(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._minimal_manifest()
            manifest["claims"] = [
                claim for claim in manifest["claims"] if claim["id"] != "cross_device_exact_parity"
            ]
            manifest_path = root / "manifest.json"
            self._write_json(manifest_path, manifest)

            report = module.verify_delivery(root, manifest_path)

            self.assertFalse(report["passed"])
            self.assertTrue(any(item["id"] == "claim:cross_device_exact_parity" for item in report["failures"]))

    def test_runtime_contract_requires_ascii_launcher_and_private_ui(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "launcher.bat").write_text("echo 禾迹", encoding="utf-8")
            (root / "ui.py").write_text("share=True", encoding="utf-8")
            manifest = self._minimal_manifest()
            manifest["runtime_contract"] = {
                "launcher": "launcher.bat",
                "launcher_ascii": True,
                "ui_source": "ui.py",
                "required_ui_tokens": [
                    "analytics_enabled=False",
                    'server_name="127.0.0.1"',
                    "share=False",
                ],
            }
            manifest_path = root / "manifest.json"
            self._write_json(manifest_path, manifest)

            report = module.verify_delivery(root, manifest_path)

            self.assertFalse(report["passed"])
            failure_ids = {item["id"] for item in report["failures"]}
            self.assertIn("runtime:launcher_ascii", failure_ids)
            self.assertIn("runtime:ui_tokens", failure_ids)

    def test_text_contract_rejects_missing_required_and_present_forbidden_claims(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "brief.md").write_text("真实机器人安全 PASS", encoding="utf-8")
            manifest = self._minimal_manifest()
            manifest["text_contracts"] = [{
                "path": "brief.md",
                "required_tokens": ["2026-09-11", "Recall 0.7143"],
                "forbidden_tokens": ["真实机器人安全 PASS"],
            }]
            manifest_path = root / "manifest.json"
            self._write_json(manifest_path, manifest)

            report = module.verify_delivery(root, manifest_path)

            self.assertFalse(report["passed"])
            failure = next(item for item in report["failures"] if item["id"] == "text:brief.md")
            self.assertEqual(failure["missing"], ["2026-09-11", "Recall 0.7143"])
            self.assertEqual(failure["forbidden_present"], ["真实机器人安全 PASS"])

    def test_rendered_board_exposes_pass_failed_blocked_and_non_claim(self):
        module = self._module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._minimal_manifest()
            manifest_path = root / "manifest.json"
            self._write_json(manifest_path, manifest)
            report = module.verify_delivery(root, manifest_path)
            report["evidence"].update({
                "crow_frozen_frames": 896,
                "crow_navigation_contract_violations": 0,
                "ssr_matched_references": 70,
                "ssr_images": 98,
                "ssr_position_mae_norm": 0.0234276,
                "ssr_heading_mae_deg": 4.373735,
            })
            board = module.render_evidence_board(report, root / "board.svg")
            svg = board.read_text(encoding="utf-8")

            self.assertIn("PASS", svg)
            self.assertIn("FAILED", svg)
            self.assertIn("BLOCKED", svg)
            self.assertIn("NOT ESTABLISHED", svg)
            self.assertIn("Offline pilot", svg)
            self.assertIn("896 frames", svg)
            self.assertIn("70/98", svg)
            self.assertIn("0.0234", svg)
            self.assertIn("4.374", svg)
            self.assertNotIn("2026-09-11", svg)
            self.assertNotIn("SAFE FOR REAL ROBOT", svg)

    def test_repository_manifest_verifies_complete_day70_package(self):
        module = self._module()
        manifest_path = DAY70_ROOT / "code" / "day70_delivery_manifest.json"

        report = module.verify_delivery(REPO_ROOT, manifest_path)

        self.assertTrue(report["passed"], report.get("failures"))
        self.assertEqual(report["marker"], "DAY70_DELIVERY_CHECK_COMPLETE")
        self.assertEqual(report["failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
