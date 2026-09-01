"""Freeze the clean RowDetr Sorghum subset as positive-only external evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from audit_downloaded_sources import validate_row_label


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rowdetr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        image_dir = args.rowdetr_root / split / "images"
        label_dir = args.rowdetr_root / split / "labels"
        for image_path in sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem)):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (720, 1280):
                continue
            label_path = label_dir / f"{image_path.stem}.json"
            row_count, issues = validate_row_label(label_path, 1280, 720)
            key = f"{split}/{image_path.stem}"
            if issues or row_count == 0:
                excluded.append({"item": key, "issues": issues or ["empty_label"]})
                continue
            records.append(
                {
                    "source_id": "rowdetr_sorghum_robot",
                    "original_split": split,
                    "item_id": image_path.stem,
                    "image_path": image_path.relative_to(args.rowdetr_root).as_posix(),
                    "label_path": label_path.relative_to(args.rowdetr_root).as_posix(),
                    "image_sha256": sha256(image_path),
                    "evidence_role": "frozen_external_test_positive_only",
                    "source_group": "rowdetr_sorghum_whole_source",
                    "sequence_group": "NOT_AVAILABLE",
                }
            )

    payload = {
        "schema_version": 1,
        "frozen_on": "2026-09-01",
        "selection_rule": "actual decoded image is 720x1280 and every row label passes coordinate QA",
        "anti_tuning_rule": "This entire source subset is evaluation-only; do not tune models, thresholds, confidence, or preprocessing from its results.",
        "scope_limit": "Positive geometry evidence only; it cannot evaluate reject behavior.",
        "sequence_group_status": "NOT_AVAILABLE",
        "record_count": len(records),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "record_count": len(records), "excluded_count": len(excluded)}, ensure_ascii=False, indent=2))
    return 0 if records and not any(record["sequence_group"] != "NOT_AVAILABLE" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
