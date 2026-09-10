"""One-time Day69 frozen evaluation. Never use its results for tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for code_dir in (
    Path(__file__).resolve().parent,
    PROJECT_ROOT / "66_crop_row_offline_video_pilot" / "code",
    PROJECT_ROOT / "68_crop_row_severe_occlusion" / "code",
):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from day66_offline_video_pilot import build_execution_predictor, load_and_verify_frozen_config  # noqa: E402
from day68_severe_occlusion import OcclusionGuardConfig  # noqa: E402
from day69_frozen_evaluation import (  # noqa: E402
    evaluate_static_entries,
    run_single_video,
    select_frozen_video_entries,
    verify_frozen_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _occlusion_config(path: Path) -> OcclusionGuardConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("marker") != "DAY68_FROZEN_CONFIG":
        raise ValueError("invalid Day68 frozen config")
    names = {item.name for item in fields(OcclusionGuardConfig)}
    return OcclusionGuardConfig(**{name: payload[name] for name in names})


def _gap_report(summary: dict[str, Any], development: dict[str, float]) -> dict[str, Any]:
    limits = {
        "row_detection_precision": ("drop", 0.10),
        "row_detection_recall": ("drop", 0.10),
        "matched_bottom_position_mae_norm": ("increase", 0.02),
        "matched_heading_mae_deg": ("increase", 3.0),
        "corridor_boundary_pair_accuracy": ("drop", 0.10),
        "corridor_center_mae_norm": ("increase", 0.02),
    }
    checks = {}
    for name, (direction, limit) in limits.items():
        test_value, dev_value = summary.get(name), development.get(name)
        if test_value is None or dev_value is None:
            continue
        gap = dev_value - test_value if direction == "drop" else test_value - dev_value
        checks[name] = {
            "development": dev_value, "test": test_value,
            "adverse_gap": gap, "maximum_adverse_gap": limit, "passed": gap <= limit,
        }
    return {"checks": checks, "all_supported_gap_gates_passed": bool(checks) and all(x["passed"] for x in checks.values())}


def run_frozen(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol_path, output_dir = Path(protocol_path), Path(output_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_evidence = verify_frozen_protocol(protocol)
    protocol_hash = _sha256(protocol_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "day69_access_ledger.json"
    if ledger_path.is_file():
        old = json.loads(ledger_path.read_text(encoding="utf-8"))
        if old.get("status") == "COMPLETE":
            raise ValueError("Day69 frozen evaluation is already COMPLETE; repeat access is forbidden")
        if old.get("protocol_sha256") != protocol_hash:
            raise ValueError("incomplete access ledger belongs to a different protocol")
    ledger = {
        "marker": "DAY69_FROZEN_ACCESS_LEDGER", "status": "STARTED",
        "protocol_sha256": protocol_hash,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "anti_tuning_rule": "No model, preprocessing, threshold, confidence, metric, or label-rule changes after this point.",
    }
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = protocol["paths"]
    crowd = json.loads(Path(paths["crow_manifest"]).read_text(encoding="utf-8"))
    episodes = select_frozen_video_entries(crowd, expected_count=int(protocol["expected_counts"]["crow_videos"]))
    temporal, optical_flow, temporal_evidence = load_and_verify_frozen_config(
        Path(paths["day66_config"]), Path(paths["day65_result"])
    )
    occlusion = _occlusion_config(Path(paths["day68_config"]))
    video_predictor = build_execution_predictor(
        Path(paths["checkpoint"]), device=protocol["execution"]["video_device"], batch_size=1
    )
    video_reports = []
    for entry in episodes:
        source = Path(entry["video_path"])
        if _sha256(source) != entry["video_sha256"]:
            raise ValueError(f"frozen video hash mismatch: {entry['episode']}")
        report = run_single_video(
            source, output_dir=output_dir / "videos" / entry["episode"],
            predictor=video_predictor, temporal_config=temporal,
            occlusion_config=occlusion, use_optical_flow=optical_flow,
            episode=entry["episode"], evidence_role=entry["role"],
        )
        expected_frames = int(entry["video"]["decoded_frame_count"])
        if report["source_frame_count"] != expected_frames:
            raise ValueError(f"frozen video decoded count mismatch: {entry['episode']}")
        report["source_sha256"] = entry["video_sha256"]
        report["predictor_runtime_ms_per_frame"] = video_predictor.last_runtime_ms_per_frame
        video_reports.append(report)

    static_predictor = build_execution_predictor(
        Path(paths["checkpoint"]), device=protocol["execution"]["static_device"],
        batch_size=int(protocol["execution"]["static_batch_size"]),
    )
    crdld_entries = _read_jsonl(Path(paths["crdld_manifest"]))
    rowdetr_manifest = json.loads(Path(paths["rowdetr_manifest"]).read_text(encoding="utf-8"))
    rowdetr_entries = rowdetr_manifest["records"]
    expected = protocol["expected_counts"]
    if len(crdld_entries) != int(expected["crdld_images"]) or len(rowdetr_entries) != int(expected["rowdetr_images"]):
        raise ValueError("static frozen manifest count mismatch")
    crdld = evaluate_static_entries(
        crdld_entries, root=Path(paths["crdld_root"]), label_kind="crdld_mask",
        predictor=static_predictor, evidence_role="same_source_internal_benchmark",
        output_jsonl=output_dir / "static" / "crdld_records.jsonl",
        batch_size=int(protocol["execution"]["static_batch_size"]),
    )
    rowdetr = evaluate_static_entries(
        rowdetr_entries, root=Path(paths["rowdetr_root"]), label_kind="rowdetr_json",
        predictor=static_predictor, evidence_role="frozen_external_test_positive_only",
        output_jsonl=output_dir / "static" / "rowdetr_records.jsonl",
        batch_size=int(protocol["execution"]["static_batch_size"]),
    )
    development = protocol["development_comparator"]
    crdld["development_gap"] = _gap_report(crdld["summary"], development)
    rowdetr["development_gap"] = _gap_report(rowdetr["summary"], development)

    video_checks = {
        "six_episodes_processed": len(video_reports) == int(expected["crow_videos"]),
        "all_overlays_decode_completely": all(x["overlay_verification"]["passed"] for x in video_reports),
        "zero_navigation_contract_violations": sum(x["navigation_invariant_violations"] for x in video_reports) == 0,
        "all_source_hashes_verified": all(x.get("source_sha256") for x in video_reports),
    }
    result = {
        "schema_version": 1,
        "marker": "DAY69_FROZEN_EVALUATION_COMPLETE",
        "protocol_sha256": protocol_hash,
        "frozen_artifacts_verified": frozen_evidence,
        "temporal_freeze_evidence": temporal_evidence,
        "video_evaluation": {
            "evidence_role": "frozen_same_source_holdout_without_framewise_corridor_truth",
            "episode_count": len(video_reports),
            "frame_count": sum(x["source_frame_count"] for x in video_reports),
            "status_counts": {
                state: sum(x["status_counts"][state] for x in video_reports)
                for state in ("valid", "candidate", "degraded", "reject")
            },
            "predictor_runtime_median_ms_per_frame": sorted(x["predictor_runtime_ms_per_frame"] for x in video_reports)[len(video_reports) // 2],
            "checks": video_checks,
            "all_engineering_checks_passed": all(video_checks.values()),
            "accuracy_and_safety_metrics": "UNAVAILABLE_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH",
            "episodes": video_reports,
        },
        "same_source_internal_benchmark": crdld,
        "frozen_external_positive_geometry": rowdetr,
        "claims": {
            "runnable_offline_pilot": all(video_checks.values()),
            "same_source_geometry_absolute_gates": crdld["summary"]["all_supported_absolute_gates_passed"],
            "external_positive_geometry_absolute_gates": rowdetr["summary"]["all_supported_absolute_gates_passed"],
            "external_generalization": "SUPPORTED_POSITIVE_GEOMETRY_ONLY" if rowdetr["summary"]["all_supported_absolute_gates_passed"] else "NOT_SUPPORTED_BY_FROZEN_EXTERNAL_POSITIVE_TEST",
            "reject_aware_external_generalization": "BLOCKED_NO_TARGET_DOMAIN_NEGATIVE_TEST",
            "real_video_safety": "BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH",
        },
        "anti_tuning_compliance": "FROZEN_RESULTS_PRESERVED_NO_POST_ACCESS_TUNING",
    }
    result_path = output_dir / "day69_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger.update({
        "status": "COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_path": str(result_path), "result_sha256": _sha256(result_path),
    })
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_frozen(args.protocol, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
