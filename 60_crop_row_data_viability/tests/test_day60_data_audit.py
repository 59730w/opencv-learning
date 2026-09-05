import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import day60_data_audit

audit_image_mask_pairs = day60_data_audit.audit_image_mask_pairs
evaluate_data_gate = day60_data_audit.evaluate_data_gate


def _write_pair(root: Path, stem: str, value: int, mask_value: int) -> None:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    mask = np.full((12, 16), mask_value, dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)
    assert cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)


def test_audit_reports_complete_decodable_pairs(tmp_path: Path) -> None:
    _write_pair(tmp_path, "a", 10, 0)
    _write_pair(tmp_path, "b", 20, 1)

    report = audit_image_mask_pairs(tmp_path)

    assert report["image_count"] == 2
    assert report["mask_count"] == 2
    assert report["missing_images"] == []
    assert report["missing_masks"] == []
    assert report["decode_failures"] == []
    assert report["shape_mismatches"] == []


def test_audit_detects_missing_pair_and_reused_mask(tmp_path: Path) -> None:
    _write_pair(tmp_path, "a", 10, 1)
    _write_pair(tmp_path, "b", 20, 1)
    (tmp_path / "masks" / "orphan.png").write_bytes(
        (tmp_path / "masks" / "a.png").read_bytes()
    )

    report = audit_image_mask_pairs(tmp_path)

    assert report["missing_images"] == ["orphan"]
    assert ["a", "b", "orphan"] in report["duplicate_mask_groups"]


def test_gate_blocks_when_frozen_external_or_provenance_is_missing() -> None:
    registry = {
        "sources": [
            {
                "id": "pilot",
                "role": "id_development",
                "available": True,
                "license_verified": True,
                "task_match": True,
                "labels_support_metrics": True,
                "provenance_groups_known": False,
                "has_required_negatives": False,
            }
        ]
    }

    result = evaluate_data_gate(registry)

    assert result["status"] == "BLOCKED"
    assert "missing_frozen_external_test" in result["blockers"]
    assert "pilot:provenance_groups_unknown" in result["blockers"]


def test_gate_passes_only_with_complete_independent_roles() -> None:
    common = {
        "available": True,
        "license_verified": True,
        "task_match": True,
        "labels_support_metrics": True,
        "provenance_groups_known": True,
        "has_required_negatives": True,
    }
    registry = {
        "sources": [
            {"id": "id", "role": "id_development", **common},
            {"id": "ood", "role": "ood_development", **common},
            {
                "id": "external",
                "role": "frozen_external_test",
                "independent_of_id": True,
                **common,
            },
        ]
    }

    result = evaluate_data_gate(registry)

    assert result == {"status": "PASS", "blockers": []}


def test_gate_ignores_unselected_research_candidates() -> None:
    common = {
        "available": True,
        "license_verified": True,
        "task_match": True,
        "labels_support_metrics": True,
        "provenance_groups_known": True,
        "has_required_negatives": True,
    }
    registry = {
        "sources": [
            {"id": "id", "role": "id_development", **common},
            {"id": "ood", "role": "ood_development", **common},
            {
                "id": "external",
                "role": "frozen_external_test",
                "independent_of_id": True,
                **common,
            },
            {
                "id": "rejected_candidate",
                "role": "candidate_only",
                "selected_for_gate": False,
            },
        ]
    }

    assert evaluate_data_gate(registry) == {"status": "PASS", "blockers": []}


def test_registry_example_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"sources": []}), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == {"sources": []}


def test_multirow_label_audit_counts_separated_rows_at_fixed_bands(tmp_path: Path) -> None:
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    label = np.zeros((100, 120), dtype=np.uint8)
    for bottom_x in (15, 45, 75, 105):
        cv2.line(label, (60, 5), (bottom_x, 99), 255, 2)
    assert cv2.imwrite(str(label_dir / "four.jpg"), label)

    assert hasattr(day60_data_audit, "audit_multirow_label_masks")
    report = day60_data_audit.audit_multirow_label_masks(label_dir)

    assert report["label_count"] == 1
    assert report["instance_ids_available"] is False
    assert report["max_rows_at_any_audit_band"] >= 4
    assert report["multirow_signal_present_fraction"] == 1.0


def test_multirow_label_audit_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="label directory"):
        day60_data_audit.audit_multirow_label_masks(tmp_path / "missing")
