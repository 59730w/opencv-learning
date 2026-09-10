"""Day69 frozen evaluation and offline crop-row pilot packaging."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for code_dir in (
    PROJECT_ROOT / "63_crop_row_geometry_extraction" / "code",
    PROJECT_ROOT / "65_crop_row_video_temporal_stability" / "code",
    PROJECT_ROOT / "66_crop_row_offline_video_pilot" / "code",
    PROJECT_ROOT / "68_crop_row_severe_occlusion" / "code",
):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from day63_crop_row_geometry import (  # noqa: E402
    CORRIDOR_AUDIT_Y_NORM,
    MULTIROW_BAND_Y_NORMS,
    CropRowLine,
    _corridor_indices,
    extract_multirow_geometry,
    match_ordered_crop_rows,
)
from day65_video_temporal import process_video_episode  # noqa: E402
from day66_offline_video_pilot import project_navigation_contract, verify_rendered_video  # noqa: E402
from day68_severe_occlusion import (  # noqa: E402
    NAVIGATION_FIELDS,
    OcclusionRecoveryGuard,
    draw_day68_overlay,
)

FROZEN_VIDEO_ROLE = "frozen_same_source_holdout"
ABSOLUTE_GATES = {
    "row_detection_precision": (">=", 0.80),
    "row_detection_recall": (">=", 0.80),
    "matched_bottom_position_mae_norm": ("<=", 0.05),
    "matched_heading_mae_deg": ("<=", 5.0),
    "corridor_boundary_pair_accuracy": (">=", 0.80),
    "corridor_center_mae_norm": ("<=", 0.05),
    "supported_valid_recall": (">=", 0.80),
    "runtime_median_ms_per_frame": ("<=", 50.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_protocol(protocol: dict[str, Any]) -> dict[str, dict[str, str]]:
    if protocol.get("marker") != "DAY69_PREACCESS_FROZEN":
        raise ValueError("missing DAY69_PREACCESS_FROZEN marker")
    artifacts = protocol.get("frozen_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("frozen_artifacts must be a non-empty object")
    evidence: dict[str, dict[str, str]] = {}
    for name, item in artifacts.items():
        path = Path(item["path"])
        if not path.is_file():
            raise ValueError(f"frozen artifact is missing: {name}: {path}")
        expected, actual = str(item["sha256"]).lower(), _sha256(path)
        if actual != expected:
            raise ValueError(
                f"frozen artifact hash mismatch for {name}: expected {expected}, got {actual}"
            )
        evidence[str(name)] = {"path": str(path), "sha256": actual}
    return evidence


def select_frozen_video_entries(
    manifest: dict[str, Any], *, expected_count: int
) -> list[dict[str, Any]]:
    selected = [dict(item) for item in manifest.get("episodes", []) if item.get("role") == FROZEN_VIDEO_ROLE]
    selected.sort(key=lambda item: str(item.get("episode", "")))
    if len(selected) != expected_count:
        raise ValueError(f"expected {expected_count} {FROZEN_VIDEO_ROLE} episodes, got {len(selected)}")
    names = [str(item.get("episode", "")) for item in selected]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("frozen video episode names must be non-empty and unique")
    if any(not item.get("video_path") for item in selected):
        raise ValueError("every frozen video entry requires video_path")
    return selected


def rowdetr_reference_rows(
    payload: dict[str, Any], *, width: int, height: int
) -> tuple[tuple[CropRowLine, ...], int]:
    """Interpolate only polylines spanning both frozen y=0.40 and y=0.90."""
    if width < 2 or height < 2:
        raise ValueError("reference dimensions must be at least 2x2")
    rows: list[CropRowLine] = []
    rejected = 0
    for label in payload.get("labels", []):
        xs = np.asarray(label.get("x", []), dtype=np.float64)
        ys = np.asarray(label.get("y", []), dtype=np.float64)
        if xs.size < 2 or xs.shape != ys.shape or not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            rejected += 1
            continue
        x_norm, y_norm = xs / (width - 1), ys / (height - 1)
        order = np.argsort(y_norm)
        unique_y, unique_indices = np.unique(y_norm[order], return_index=True)
        unique_x = x_norm[order][unique_indices]
        if unique_y.size < 2 or unique_y[0] > 0.40 or unique_y[-1] < 0.90:
            rejected += 1
            continue
        far = float(np.interp(0.40, unique_y, unique_x))
        near = float(np.interp(0.90, unique_y, unique_x))
        support = sum(unique_y[0] <= band <= unique_y[-1] for band in MULTIROW_BAND_Y_NORMS)
        rows.append(CropRowLine(far, near, 1.0, int(support)))
    # Day63's frozen matcher defines row identity by the far anchor position.
    rows.sort(key=lambda row: row.far_x_norm)
    return tuple(rows), rejected


def _gate_pass(value: float | None, operator: str, threshold: float) -> bool | None:
    if value is None:
        return None
    return value >= threshold if operator == ">=" else value <= threshold


def geometry_summary(records: Sequence[dict[str, Any]], *, evidence_role: str) -> dict[str, Any]:
    predicted = sum(int(row.get("predicted_count", 0)) for row in records)
    reference = sum(int(row.get("reference_count", 0)) for row in records)
    matched = sum(int(row.get("matched_count", 0)) for row in records)
    ref_corridor = [row for row in records if row.get("reference_corridor_available")]
    paired = [row for row in ref_corridor if row.get("prediction_corridor_available")]
    center_errors = [float(row["corridor_center_error_norm"]) for row in paired if row.get("corridor_center_error_norm") is not None]
    runtimes = [float(row["runtime_ms"]) for row in records if row.get("runtime_ms") is not None]
    values: dict[str, float | None] = {
        "row_detection_precision": matched / predicted if predicted else (1.0 if not reference else 0.0),
        "row_detection_recall": matched / reference if reference else (1.0 if not predicted else 0.0),
        "matched_bottom_position_mae_norm": sum(float(row.get("position_error_sum", 0.0)) for row in records) / matched if matched else None,
        "matched_heading_mae_deg": sum(float(row.get("heading_error_sum", 0.0)) for row in records) / matched if matched else None,
        "corridor_boundary_pair_accuracy": sum(bool(row.get("boundary_pair_correct")) for row in ref_corridor) / len(ref_corridor) if ref_corridor else None,
        "corridor_center_mae_norm": statistics.fmean(center_errors) if center_errors else None,
        "supported_valid_recall": sum(row.get("prediction_status") == "valid" for row in ref_corridor) / len(ref_corridor) if ref_corridor else None,
        "runtime_median_ms_per_frame": statistics.median(runtimes) if runtimes else None,
    }
    gates = {
        name: {"value": values[name], "operator": op, "threshold": threshold, "passed": _gate_pass(values[name], op, threshold)}
        for name, (op, threshold) in ABSOLUTE_GATES.items()
        if values[name] is not None
    }
    return {
        "evidence_role": evidence_role,
        "record_count": len(records),
        "predicted_row_count": predicted,
        "reference_row_count": reference,
        "matched_row_count": matched,
        **values,
        "unavailable_metrics": ["unsafe_false_valid_rate", "reject_precision", "reject_recall", "target_domain_safety"],
        "gates": gates,
        "all_supported_absolute_gates_passed": bool(gates) and all(item["passed"] is True for item in gates.values()),
    }


def evaluate_static_entries(
    entries: Sequence[dict[str, Any]],
    *,
    root: Path,
    label_kind: str,
    predictor: Any,
    evidence_role: str,
    output_jsonl: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate hash-verified positive geometry images with frozen definitions."""
    if label_kind not in {"crdld_mask", "rowdetr_json"}:
        raise ValueError(f"unsupported label kind: {label_kind}")
    root, output_jsonl = Path(root), Path(output_jsonl)
    records: list[dict[str, Any]] = []
    verified = 0
    for start in range(0, len(entries), batch_size):
        batch_entries = list(entries[start : start + batch_size])
        frames: list[np.ndarray] = []
        references: list[tuple[tuple[CropRowLine, ...], int]] = []
        for entry in batch_entries:
            image_path = root / entry["image_path"]
            expected, actual = str(entry["image_sha256"]).lower(), _sha256(image_path)
            if actual != expected:
                raise ValueError(f"image hash mismatch for {entry.get('item_id')}: expected {expected}, got {actual}")
            verified += 1
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode image: {image_path}")
            label_path = root / entry["label_path"]
            if label_kind == "crdld_mask":
                label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                if label is None:
                    raise ValueError(f"cannot decode label: {label_path}")
                binary = np.where(label > 127, 255, 0).astype(np.uint8)
                reference = extract_multirow_geometry(binary, label_mode=True).rows
                references.append((reference, 0))
            else:
                payload = json.loads(label_path.read_text(encoding="utf-8"))
                references.append(rowdetr_reference_rows(payload, width=image.shape[1], height=image.shape[0]))
            frames.append(image)
        predictions = predictor.predict(frames)
        if len(predictions) != len(batch_entries):
            raise ValueError("predictor must return one row sequence per image")
        runtime = float(getattr(predictor, "last_runtime_ms_per_frame", 0.0) or 0.0)
        for entry, predicted, (reference, rejected) in zip(batch_entries, predictions, references):
            pred_rows = tuple(sorted(predicted, key=lambda row: row.far_x_norm))
            ref_rows = tuple(sorted(reference, key=lambda row: row.far_x_norm))
            matching = match_ordered_crop_rows(pred_rows, ref_rows)
            mapping = {pred_index: ref_index for pred_index, ref_index in matching["pairs"]}
            pred_pair, ref_pair = _corridor_indices(pred_rows), _corridor_indices(ref_rows)
            pair_correct = bool(
                pred_pair is not None and ref_pair is not None
                and mapping.get(pred_pair[0]) == ref_pair[0]
                and mapping.get(pred_pair[1]) == ref_pair[1]
            )
            center_error = None
            if pred_pair is not None and ref_pair is not None:
                pred_center = (pred_rows[pred_pair[0]].near_x_norm + pred_rows[pred_pair[1]].near_x_norm) / 2.0
                ref_center = (ref_rows[ref_pair[0]].near_x_norm + ref_rows[ref_pair[1]].near_x_norm) / 2.0
                center_error = abs(pred_center - ref_center)
            records.append({
                "item_id": str(entry.get("item_id")),
                "evidence_role": evidence_role,
                "predicted_count": matching["predicted_count"],
                "reference_count": matching["reference_count"],
                "matched_count": matching["matched_count"],
                "position_error_sum": (matching["position_mae_norm"] or 0.0) * matching["matched_count"],
                "heading_error_sum": (matching["heading_mae_deg"] or 0.0) * matching["matched_count"],
                "reference_corridor_available": ref_pair is not None,
                "prediction_corridor_available": pred_pair is not None,
                "boundary_pair_correct": pair_correct,
                "corridor_center_error_norm": center_error,
                "prediction_status": "valid" if pred_pair is not None else ("degraded" if pred_rows else "reject"),
                "reference_polylines_not_evaluable": rejected,
                "runtime_ms": runtime,
            })
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, records)
    return {
        "evidence_role": evidence_role,
        "label_kind": label_kind,
        "image_hashes_verified": verified,
        "output_jsonl": str(output_jsonl),
        "records_sha256": _sha256(output_jsonl),
        "summary": geometry_summary(records, evidence_role=evidence_role),
    }
def _navigation_invariant_violation(record: dict[str, Any]) -> bool:
    available = bool(record.get("navigation_available"))
    valid = record.get("pilot_state") == "valid" and record.get("status") == "valid"
    complete = all(record.get(field) is not None for field in NAVIGATION_FIELDS)
    return available != valid or available != complete


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def run_single_video(
    video_path: Path,
    *,
    output_dir: Path,
    predictor: Any,
    temporal_config: Any,
    occlusion_config: Any,
    use_optical_flow: bool,
    episode: str,
    evidence_role: str,
) -> dict[str, Any]:
    """Run the frozen Day63->65->66->68 stack and emit aligned artifacts."""
    video_path, output_dir = Path(video_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records, temporal_summary = process_video_episode(
        video_path, predictor=predictor, config=temporal_config,
        episode=episode, use_optical_flow=use_optical_flow,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot reopen video for rendering: {video_path}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 10.0
    overlay_path = output_dir / f"{episode}_overlay.mp4"
    render_size = (width * 2, height * 2)
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, render_size)
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"cannot create overlay video: {overlay_path}")
    guard, records = OcclusionRecoveryGuard(occlusion_config), []
    try:
        for raw in raw_records:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("source video ended before temporal records")
            projected = project_navigation_contract(raw)
            projected["evidence_role"] = evidence_role
            guarded = guard.apply(projected, frame)
            records.append(guarded)
            display = cv2.resize(frame, render_size, interpolation=cv2.INTER_LINEAR)
            writer.write(draw_day68_overlay(display, guarded))
        extra, _ = capture.read()
        if extra:
            raise ValueError("source video contains more frames than temporal records")
    finally:
        capture.release()
        writer.release()

    jsonl_path = output_dir / f"{episode}_frames.jsonl"
    csv_path = output_dir / f"{episode}_frames.csv"
    report_path = output_dir / f"{episode}_report.json"
    _write_jsonl(jsonl_path, records)
    fieldnames = [
        "episode", "frame_index", "evidence_role", "pilot_state", "status", "navigation_available",
        "confidence", "lateral_offset_norm", "heading_error_deg", "row_spacing_norm", "crop_rows",
        "corridor_left_boundary", "corridor_right_boundary", "corridor_centerline_points_norm",
        "vanishing_point_norm", "reason", "diagnostics",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        csv_writer.writeheader()
        for row in records:
            csv_writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    result = {
        "marker": "DAY69_SINGLE_VIDEO_COMPLETE", "episode": episode, "evidence_role": evidence_role,
        "source_video": str(video_path), "source_frame_count": len(records), "temporal_summary": temporal_summary,
        "status_counts": {state: sum(row["pilot_state"] == state for row in records) for state in ("valid", "candidate", "degraded", "reject")},
        "navigation_invariant_violations": sum(_navigation_invariant_violation(row) for row in records),
        "output_jsonl": str(jsonl_path), "output_csv": str(csv_path), "overlay_video": str(overlay_path),
        "report_path": str(report_path),
        "overlay_verification": verify_rendered_video(overlay_path, expected_frames=len(records)),
    }
    result["artifacts_sha256"] = {"jsonl": _sha256(jsonl_path), "csv": _sha256(csv_path), "overlay_video": _sha256(overlay_path)}
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
