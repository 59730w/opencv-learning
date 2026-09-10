"""Build deterministic Day69 overlay contact sheets for human inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (str(row["episode"]), int(row["frame_index"])))
    if len(rows) <= count:
        return rows
    indices = sorted({round(i * (len(rows) - 1) / (count - 1)) for i in range(count)})
    return [rows[index] for index in indices]


def build_audit(result_path: Path, output_dir: Path, *, per_group: int = 8) -> dict[str, Any]:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if result.get("marker") != "DAY69_FROZEN_EVALUATION_COMPLETE":
        raise ValueError("expected a completed Day69 frozen result")
    reports = result["video_evaluation"]["episodes"]
    records, overlay_by_episode = [], {}
    for report in reports:
        episode = report["episode"]
        overlay_by_episode[episode] = Path(report["overlay_video"])
        records.extend(
            json.loads(line) for line in Path(report["output_jsonl"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    groups = {state: [row for row in records if row["pilot_state"] == state] for state in ("valid", "candidate", "degraded", "reject")}
    previous = {}
    transitions = []
    for row in sorted(records, key=lambda x: (str(x["episode"]), int(x["frame_index"]))):
        old = previous.get(row["episode"])
        if old is not None and old != row["pilot_state"]:
            transitions.append(row)
        previous[row["episode"]] = row["pilot_state"]
    groups["transition"] = transitions
    groups["day68_guard"] = [row for row in records if (row.get("diagnostics") or {}).get("day68_guard_triggered")]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, sheets = {}, {}
    for name, candidates in groups.items():
        selected = _evenly_spaced(candidates, per_group)
        metadata[name] = [{"episode": row["episode"], "frame_index": row["frame_index"], "pilot_state": row["pilot_state"], "reason": row.get("reason")} for row in selected]
        frames = []
        for row in selected:
            capture = cv2.VideoCapture(str(overlay_by_episode[row["episode"]]))
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise ValueError(f"cannot decode audit frame {row['episode']}:{row['frame_index']}")
            frames.append(frame)
        if not frames:
            continue
        height, width = frames[0].shape[:2]
        frames = [frame if frame.shape[:2] == (height, width) else cv2.resize(frame, (width, height)) for frame in frames]
        blank = np.zeros_like(frames[0])
        rows_of_images = []
        for start in range(0, len(frames), 2):
            pair = frames[start : start + 2]
            pair.extend([blank.copy()] * (2 - len(pair)))
            rows_of_images.append(cv2.hconcat(pair))
        sheet_path = output_dir / f"day69_audit_{name}.jpg"
        if not cv2.imwrite(str(sheet_path), cv2.vconcat(rows_of_images)):
            raise ValueError(f"cannot write audit sheet: {sheet_path}")
        sheets[name] = str(sheet_path)
    audit = {
        "marker": "DAY69_VISUAL_AUDIT_READY",
        "selection": "deterministic evenly spaced frames per state, transition, and Day68 guard group",
        "per_group": per_group,
        "selected": metadata,
        "contact_sheets": sheets,
        "evidence_boundary": "Visual inspection detects obvious rendering/geometry failures; it is not independent framewise safety ground truth.",
    }
    audit_path = output_dir / "day69_visual_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit["audit_path"] = str(audit_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build_audit(args.result, args.output_dir, per_group=args.per_group), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
