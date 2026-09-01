from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REQUIRED_METRICS = {
    "bottom_position_mae_norm",
    "heading_mae_deg",
    "supported_valid_recall",
    "unsafe_false_valid_rate",
    "runtime_median_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Day59 learning contract")
    parser.add_argument("--lesson-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise AssertionError(f"{path} must contain a mapping")
    return loaded


def main() -> None:
    args = parse_args()
    contract_path = args.project_root / "target_contract.yaml"
    registry_path = args.project_root / "evidence_registry.yaml"
    review_path = args.project_root / "docs" / "open_source_baseline_review.md"
    notes_path = args.lesson_root / "code" / "day59_notes.md"

    for path in (contract_path, registry_path, review_path, notes_path):
        assert path.is_file(), f"missing required Day59 file: {path}"

    contract = load_yaml(contract_path)
    registry = load_yaml(registry_path)

    assert contract["gate"]["verdict"] == "PASS"
    assert contract["project"]["id"] == "04_crop_row_perception"
    assert contract["output_contract"]["coordinate_system"]["pixel_origin"] == "top_left"
    assert contract["output_contract"]["status"]["enum"] == ["valid", "degraded", "reject"]
    metric_ids = {item["id"] for item in contract["metrics_and_thresholds"]}
    assert metric_ids == REQUIRED_METRICS
    external_gate = contract["acceptance_rules"]["frozen_external_gate"]
    assert external_gate["independent_source_required"] is True
    assert external_gate["same_absolute_thresholds_as_internal"] is True
    assert external_gate["gap_limits"]["bottom_position_mae_norm"] == "<= 0.02"
    assert external_gate["gap_limits"]["heading_mae_deg"] == "<= 3.0"
    assert external_gate["gap_limits"]["unsafe_false_valid_rate"] == "<= 0.03"
    assert "external_threshold_failure" in external_gate["blocking_conditions"]

    roles = {item["role"]: item["status"] for item in registry["entries"]}
    assert roles["background_reference"] == "AVAILABLE"
    assert roles["id_development"] in {"NOT_AVAILABLE", "BLOCKED"}
    assert roles["ood_development"] in {"NOT_AVAILABLE", "BLOCKED"}
    assert roles["frozen_external_test"] in {"NOT_AVAILABLE", "BLOCKED"}

    review = review_path.read_text(encoding="utf-8")
    assert review.count("https://github.com/") >= 7
    assert "不复制代码" in review

    notes = notes_path.read_text(encoding="utf-8")
    assert "DAY59_TARGET_CONTRACT_PASS" in notes
    assert "首要待审查数据候选" in notes
    assert "尚未下载或实测" in notes
    assert "不能保证内部和外部结果都好" in notes

    print("DAY59_TARGET_CONTRACT_PASS")
    print("DAY59_CONTRACT_OK")


if __name__ == "__main__":
    main()
