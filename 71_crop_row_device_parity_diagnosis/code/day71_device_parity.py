"""Day71 layered CPU/CUDA parity diagnosis for the frozen crop-row pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = Path(
    r"D:\DL_code\data\crop_row_perception\sources\lecrop_data\rgb_episodes"
    r"\crowfollow_lecropfollow_exp_2026-01-26-12-35-02_ep0.mp4"
)
DEFAULT_CHECKPOINT = Path(
    r"D:\DL_code\data\crop_row_perception\day63_crop_row_geometry"
    r"\day63_resnet18_centerline_model.pt"
)
DEFAULT_DAY65_RESULT = Path(
    r"D:\DL_code\data\crop_row_perception\day65_video_temporal_verified"
    r"\day65_results.json"
)
DEFAULT_TEMPORAL_CONFIG = (
    PROJECT_ROOT / "66_crop_row_offline_video_pilot" / "code" / "day66_frozen_config.json"
)
DEFAULT_OCCLUSION_CONFIG = (
    PROJECT_ROOT / "68_crop_row_severe_occlusion" / "code" / "day68_frozen_config.json"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\DL_code\data\crop_row_perception\day71_device_parity"
)
DEFAULT_REPORT = Path(__file__).with_name("day71_device_parity_result.json")
DEFAULT_DIFFERENCES_CSV = Path(__file__).with_name("day71_frame_differences.csv")
DEFAULT_CPU_CLI_JSONL = (
    DEFAULT_OUTPUT_DIR
    / "production_cpu_cli_run"
    / "crowfollow_lecropfollow_exp_2026-01-26-12-35-02_ep0"
    / "crowfollow_lecropfollow_exp_2026-01-26-12-35-02_ep0_frames.jsonl"
)
DEFAULT_STABLE_CLI_JSONL = (
    DEFAULT_OUTPUT_DIR
    / "stable_cli_run"
    / "crowfollow_lecropfollow_exp_2026-01-26-12-35-02_ep0"
    / "crowfollow_lecropfollow_exp_2026-01-26-12-35-02_ep0_frames.jsonl"
)
DEFAULT_CPU_CLI_REPORT = DEFAULT_OUTPUT_DIR / "production_cpu_cli_run" / "pilot_report.json"
DEFAULT_STABLE_CLI_REPORT = DEFAULT_OUTPUT_DIR / "stable_cli_run" / "pilot_report.json"
EXPECTED_DAY70_VIDEO_SHA256 = (
    "b9da6a1575659cdf8478764736fbf26bf14b7aed4b69bafa20d2b7b4a0095968"
)


def array_fingerprint(array: np.ndarray) -> str:
    """Hash values together with dtype and shape so evidence is unambiguous."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def compare_probability_stacks(
    left: np.ndarray, right: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    """Summarize numeric and threshold-level differences between two runs."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("probability stacks must have the same frame-first shape")
    absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
    crossings = np.not_equal(left >= threshold, right >= threshold)
    per_frame = crossings.reshape(len(left), -1).sum(axis=1)
    return {
        "frame_count": int(len(left)),
        "exactly_equal": bool(np.array_equal(left, right)),
        "max_abs_difference": float(absolute.max(initial=0.0)),
        "mean_abs_difference": float(absolute.mean()) if absolute.size else 0.0,
        "threshold_crossing_pixels": int(crossings.sum()),
        "frames_with_threshold_crossings": int(np.count_nonzero(per_frame)),
        "first_threshold_crossing_frame": (
            int(np.flatnonzero(per_frame)[0]) if np.any(per_frame) else None
        ),
    }


def _contiguous_runs(indices: Sequence[int]) -> list[list[int]]:
    if not indices:
        return []
    runs: list[list[int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current != previous + 1:
            runs.append([start, previous])
            start = current
        previous = current
    runs.append([start, previous])
    return runs


def compare_final_records(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Locate final pilot-state and navigation differences frame by frame."""
    if len(left) != len(right):
        raise ValueError("record sequences must have equal length")
    status_indices = [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if a.get("pilot_state") != b.get("pilot_state")
    ]
    navigation_indices = [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if bool(a.get("navigation_available"))
        != bool(b.get("navigation_available"))
    ]
    return {
        "frame_count": len(left),
        "status_mismatch_frames": len(status_indices),
        "navigation_mismatch_frames": len(navigation_indices),
        "first_status_mismatch_frame": status_indices[0] if status_indices else None,
        "first_navigation_mismatch_frame": (
            navigation_indices[0] if navigation_indices else None
        ),
        "status_mismatch_indices": status_indices,
        "navigation_mismatch_indices": navigation_indices,
        "status_mismatch_runs": _contiguous_runs(status_indices),
    }


def classify_root_cause(comparisons: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Classify only factors isolated by the three planned controlled contrasts."""
    no_tf32_bs1_key = "cpu_eager_bs1__cuda_no_tf32_eager_bs1"
    no_tf32_bs32_key = "cpu_eager_bs1__cuda_no_tf32_eager_bs32"
    direct_tf32_bs32_key = "cuda_eager_bs32__cuda_no_tf32_eager_bs32"
    if (
        no_tf32_bs1_key in comparisons
        and no_tf32_bs32_key in comparisons
        and comparisons[no_tf32_bs1_key]["status_mismatch_frames"] == 0
        and comparisons[no_tf32_bs32_key]["status_mismatch_frames"] == 0
        and comparisons["cpu_eager_bs1__cuda_eager_bs1"][
            "status_mismatch_frames"
        ]
        > 0
        and direct_tf32_bs32_key in comparisons
        and comparisons[direct_tf32_bs32_key]["status_mismatch_frames"] > 0
    ):
        return {
            "status": "ROOT_CAUSE_IDENTIFIED",
            "primary_factor": "cudnn_tf32_execution_path",
            "reason": "Disabling only cuDNN TF32 removed final-state differences at both CUDA batch sizes on the development audit video.",
        }
    optimized = comparisons["cpu_eager_bs1__cpu_optimized_bs1"][
        "status_mismatch_frames"
    ]
    device = comparisons["cpu_eager_bs1__cuda_eager_bs1"][
        "status_mismatch_frames"
    ]
    batch = comparisons["cuda_eager_bs1__cuda_eager_bs32"][
        "status_mismatch_frames"
    ]
    active = [
        name
        for name, count in (
            ("cpu_optimized_backend", optimized),
            ("device_numeric_path", device),
            ("cuda_batch_shape", batch),
        )
        if count
    ]
    if len(active) == 1:
        return {
            "status": "ROOT_CAUSE_IDENTIFIED",
            "primary_factor": active[0],
            "reason": "One controlled contrast isolated the only final-state difference.",
        }
    if active:
        return {
            "status": "PARTIALLY_IDENTIFIED",
            "primary_factor": "+".join(active),
            "reason": "Multiple controlled contrasts differ; contributions are not uniquely isolated.",
        }
    return {
        "status": "UNRESOLVED",
        "primary_factor": "none_at_final_state",
        "reason": "No controlled contrast reproduced a final-state difference.",
    }


def recommend_execution_policy(
    comparisons: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Prefer the smallest measured execution-only change that restores parity."""
    no_tf32_bs32_key = "cpu_eager_bs1__cuda_no_tf32_eager_bs32"
    production_no_tf32_bs32_key = (
        "cpu_optimized_bs1__cuda_no_tf32_eager_bs32"
    )
    if (
        no_tf32_bs32_key in comparisons
        and comparisons[no_tf32_bs32_key]["status_mismatch_frames"] == 0
        and (
            production_no_tf32_bs32_key not in comparisons
            or comparisons[production_no_tf32_bs32_key]["status_mismatch_frames"]
            == 0
        )
    ):
        return {
            "status": "PASS_DEVELOPMENT_REPRODUCIBILITY",
            "policy": "cuda_no_tf32_eager_batch32",
            "changes_model_or_thresholds": False,
            "reason": "Disabling cuDNN TF32 retained batch-32 throughput while removing final-state differences on the development audit video.",
        }
    no_tf32_key = "cpu_eager_bs1__cuda_no_tf32_eager_bs1"
    if (
        no_tf32_key in comparisons
        and comparisons[no_tf32_key]["status_mismatch_frames"] == 0
    ):
        return {
            "status": "PASS_DEVELOPMENT_REPRODUCIBILITY",
            "policy": "cuda_no_tf32_eager_batch1",
            "changes_model_or_thresholds": False,
            "reason": "Disabling cuDNN TF32 with eager batch-1 removed final-state differences on the development audit video.",
        }
    matched = comparisons["cpu_eager_bs1__cuda_eager_bs1"][
        "status_mismatch_frames"
    ]
    current = comparisons["cpu_optimized_bs1__cuda_eager_bs32"][
        "status_mismatch_frames"
    ]
    if current and not matched:
        return {
            "status": "PASS_DEVELOPMENT_REPRODUCIBILITY",
            "policy": "matched_eager_batch1",
            "changes_model_or_thresholds": False,
            "reason": "The matched eager batch-1 contrast removed final-state differences on the development audit video.",
        }
    return {
        "status": "FIXED_DEVICE_REQUIRED",
        "policy": "explicit_cpu_reference",
        "changes_model_or_thresholds": False,
        "reason": "Matched execution did not establish cross-device final-state parity; keep an explicit device contract.",
    }


def run_model_probabilities(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: str,
    batch_size: int,
    channels_last: bool,
) -> np.ndarray:
    """Run sigmoid inference with an explicit device, memory format, and batch."""
    if features.ndim != 4 or features.shape[1] != 4:
        raise ValueError("features must have shape [N, 4, H, W]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for index in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[index : index + batch_size]).to(
                device=device, dtype=torch.float32
            )
            batch = batch.div(255.0)
            if channels_last:
                batch = batch.contiguous(memory_format=torch.channels_last)
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy()[:, 0])
    if not probabilities:
        return np.empty((0, features.shape[2], features.shape[3]), dtype=np.float32)
    return np.concatenate(probabilities, axis=0)


def compare_observations(
    left: Sequence[Sequence[dict[str, Any]]],
    right: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare already ordered decoded rows without hiding count differences."""
    if len(left) != len(right):
        raise ValueError("observation sequences must have equal length")
    count_mismatches: list[int] = []
    near_differences: list[float] = []
    far_differences: list[float] = []
    confidence_differences: list[float] = []
    for frame_index, (left_rows, right_rows) in enumerate(zip(left, right)):
        if len(left_rows) != len(right_rows):
            count_mismatches.append(frame_index)
        for left_row, right_row in zip(left_rows, right_rows):
            near_differences.append(
                abs(float(left_row["near_x_norm"]) - float(right_row["near_x_norm"]))
            )
            far_differences.append(
                abs(float(left_row["far_x_norm"]) - float(right_row["far_x_norm"]))
            )
            confidence_differences.append(
                abs(float(left_row["confidence"]) - float(right_row["confidence"]))
            )
    return {
        "frame_count": len(left),
        "row_count_mismatch_frames": len(count_mismatches),
        "row_count_mismatch_indices": count_mismatches,
        "first_row_count_mismatch_frame": (
            count_mismatches[0] if count_mismatches else None
        ),
        "max_matched_near_x_abs_difference": max(near_differences, default=0.0),
        "max_matched_far_x_abs_difference": max(far_differences, default=0.0),
        "max_matched_confidence_abs_difference": max(
            confidence_differences, default=0.0
        ),
    }


def _compare_geometry_values(left: Any, right: Any) -> tuple[bool, float]:
    """Return (same structure/categorical values, maximum numeric drift)."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right, 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return True, abs(float(left) - float(right))
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False, 0.0
        comparisons = [_compare_geometry_values(a, b) for a, b in zip(left, right)]
        return all(item[0] for item in comparisons), max(
            (item[1] for item in comparisons), default=0.0
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False, 0.0
        comparisons = [_compare_geometry_values(left[key], right[key]) for key in left]
        return all(item[0] for item in comparisons), max(
            (item[1] for item in comparisons), default=0.0
        )
    return left == right, 0.0


def compare_emitted_jsonl(
    left_path: Path,
    right_path: Path,
    *,
    geometry_abs_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare real CLI output: exact decisions plus bounded float geometry."""
    paths = (Path(left_path), Path(right_path))
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing emitted JSONL: {path}")
    records = [
        [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
        for path in paths
    ]
    summary = compare_final_records(records[0], records[1])
    identity_indices = [
        index
        for index, (left, right) in enumerate(zip(records[0], records[1]))
        if left.get("frame_index") != right.get("frame_index")
        or left.get("episode") != right.get("episode")
    ]
    geometry_fields = (
        "status",
        "confidence",
        "lateral_offset_norm",
        "heading_error_deg",
        "row_spacing_norm",
        "crop_rows",
        "corridor_left_boundary",
        "corridor_right_boundary",
        "corridor_centerline_points_norm",
        "vanishing_point_norm",
    )
    geometry_indices = []
    structural_indices = []
    exact_difference_indices = []
    max_abs_difference = 0.0
    for index, (left, right) in enumerate(zip(records[0], records[1])):
        structure_matches = True
        frame_max_abs_difference = 0.0
        frame_exactly_equal = True
        for field in geometry_fields:
            left_value, right_value = left.get(field), right.get(field)
            if left_value != right_value:
                frame_exactly_equal = False
            field_structure_matches, field_max_abs_difference = (
                _compare_geometry_values(left_value, right_value)
            )
            structure_matches = structure_matches and field_structure_matches
            frame_max_abs_difference = max(
                frame_max_abs_difference, field_max_abs_difference
            )
        max_abs_difference = max(max_abs_difference, frame_max_abs_difference)
        if not frame_exactly_equal:
            exact_difference_indices.append(index)
        if not structure_matches:
            structural_indices.append(index)
        if not structure_matches or frame_max_abs_difference > geometry_abs_tolerance:
            geometry_indices.append(index)
    return {
        **summary,
        "frame_identity_mismatch_frames": len(identity_indices),
        "frame_identity_mismatch_indices": identity_indices,
        "geometry_mismatch_frames": len(geometry_indices),
        "geometry_mismatch_indices": geometry_indices,
        "geometry_structural_mismatch_frames": len(structural_indices),
        "geometry_structural_mismatch_indices": structural_indices,
        "geometry_exact_difference_frames": len(exact_difference_indices),
        "geometry_max_abs_difference": max_abs_difference,
        "geometry_abs_tolerance": geometry_abs_tolerance,
        "left_jsonl": str(paths[0].resolve()),
        "right_jsonl": str(paths[1].resolve()),
    }


def validate_cli_provenance(
    *,
    video_path: Path,
    cpu_jsonl: Path,
    cpu_report: Path,
    stable_jsonl: Path,
    stable_report: Path,
) -> dict[str, Any]:
    """Bind compared JSONLs to their Day69 reports and Day71 execution policy."""
    paths = {
        "video": Path(video_path),
        "cpu_jsonl": Path(cpu_jsonl),
        "cpu_report": Path(cpu_report),
        "stable_jsonl": Path(stable_jsonl),
        "stable_report": Path(stable_report),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing CLI provenance {label}: {path}")
    cpu_payload = json.loads(paths["cpu_report"].read_text(encoding="utf-8-sig"))
    stable_payload = json.loads(
        paths["stable_report"].read_text(encoding="utf-8-sig")
    )
    cpu_records = [
        json.loads(line)
        for line in paths["cpu_jsonl"].read_text(encoding="utf-8-sig").splitlines()
    ]
    stable_records = [
        json.loads(line)
        for line in paths["stable_jsonl"].read_text(encoding="utf-8-sig").splitlines()
    ]
    cpu_runs = cpu_payload.get("reports", [])
    stable_runs = stable_payload.get("reports", [])
    cpu_run = cpu_runs[0] if len(cpu_runs) == 1 else {}
    stable_run = stable_runs[0] if len(stable_runs) == 1 else {}
    policy = stable_payload.get("day71_execution_policy", {})

    def same_path(left: Any, right: Path) -> bool:
        try:
            return Path(left).resolve() == right.resolve()
        except (TypeError, OSError):
            return False

    def artifact_hash_matches(run: dict[str, Any], path: Path) -> bool:
        expected = run.get("artifacts_sha256", {}).get("jsonl")
        return expected == hashlib.sha256(path.read_bytes()).hexdigest()

    cpu_episode = cpu_run.get("episode")
    stable_episode = stable_run.get("episode")
    checks = {
        "reports_are_day69_outputs": cpu_payload.get("marker")
        == stable_payload.get("marker")
        == "DAY69_OFFLINE_PILOT_COMPLETE",
        "reports_bind_same_input_video": same_path(cpu_payload.get("input"), paths["video"])
        and same_path(stable_payload.get("input"), paths["video"])
        and same_path(cpu_run.get("source_video"), paths["video"])
        and same_path(stable_run.get("source_video"), paths["video"]),
        "reports_bind_compared_jsonl_paths": same_path(
            cpu_run.get("output_jsonl"), paths["cpu_jsonl"]
        )
        and same_path(stable_run.get("output_jsonl"), paths["stable_jsonl"]),
        "reports_bind_compared_jsonl_hashes": artifact_hash_matches(
            cpu_run, paths["cpu_jsonl"]
        )
        and artifact_hash_matches(stable_run, paths["stable_jsonl"]),
        "reports_and_jsonl_frame_counts_match": cpu_payload.get("frame_count")
        == len(cpu_records)
        and stable_payload.get("frame_count") == len(stable_records),
        "jsonl_frame_indices_are_sequential": [
            record.get("frame_index") for record in cpu_records
        ]
        == list(range(len(cpu_records)))
        and [record.get("frame_index") for record in stable_records]
        == list(range(len(stable_records))),
        "reports_and_jsonl_episodes_match": cpu_episode == stable_episode
        and all(record.get("episode") == cpu_episode for record in cpu_records)
        and all(record.get("episode") == stable_episode for record in stable_records),
        "cpu_report_uses_production_backend": cpu_payload.get("device") == "cpu"
        and cpu_payload.get("predictor_backend")
        == "torchscript_optimize_for_inference_channels_last",
        "stable_report_uses_validated_policy": stable_payload.get("device") == "cuda"
        and stable_payload.get("predictor_backend") == "torch_eager"
        and policy.get("policy") == "cuda_no_tf32_eager_batch32"
        and policy.get("effective_device") == "cuda"
        and policy.get("effective_batch_size") == 32
        and policy.get("cudnn_allow_tf32") is False,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "cpu_report": str(paths["cpu_report"].resolve()),
        "stable_report": str(paths["stable_report"].resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose CPU/CUDA divergence in the frozen crop-row pilot."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--day65-result", type=Path, default=DEFAULT_DAY65_RESULT)
    parser.add_argument("--temporal-config", type=Path, default=DEFAULT_TEMPORAL_CONFIG)
    parser.add_argument("--occlusion-config", type=Path, default=DEFAULT_OCCLUSION_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--differences-csv", type=Path, default=DEFAULT_DIFFERENCES_CSV)
    parser.add_argument("--cpu-cli-jsonl", type=Path, default=DEFAULT_CPU_CLI_JSONL)
    parser.add_argument("--stable-cli-jsonl", type=Path, default=DEFAULT_STABLE_CLI_JSONL)
    parser.add_argument("--cpu-cli-report", type=Path, default=DEFAULT_CPU_CLI_REPORT)
    parser.add_argument("--stable-cli-report", type=Path, default=DEFAULT_STABLE_CLI_REPORT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--peak-height", type=float, default=0.20)
    return parser


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    """Run six controlled execution arms while keeping the frozen stack intact."""
    required = {
        "video": Path(args.video),
        "checkpoint": Path(args.checkpoint),
        "day65 result": Path(args.day65_result),
        "temporal config": Path(args.temporal_config),
        "occlusion config": Path(args.occlusion_config),
    }
    for label, path in required.items():
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")
    if args.repeats < 2:
        raise ValueError("repeats must be at least two")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for the six-arm Day71 diagnosis")

    code_dirs = (
        PROJECT_ROOT / "63_crop_row_geometry_extraction" / "code",
        PROJECT_ROOT / "65_crop_row_video_temporal_stability" / "code",
        PROJECT_ROOT / "66_crop_row_offline_video_pilot" / "code",
        PROJECT_ROOT / "68_crop_row_severe_occlusion" / "code",
    )
    for code_dir in code_dirs:
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

    import cv2
    from day63_crop_row_geometry import decode_centerline_heatmap
    from day65_video_temporal import FrozenDay63Predictor, process_video_episode, prepare_video_feature
    from day66_offline_video_pilot import (
        OptimizedCpuFrozenDay63Predictor,
        load_and_verify_frozen_config,
        project_navigation_contract,
    )
    from day68_severe_occlusion import (
        OcclusionGuardConfig,
        apply_guard_to_records,
    )

    video_path = Path(args.video).resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames_bgr.append(frame)
    capture.release()
    if not frames_bgr:
        raise ValueError("video decoded zero frames")

    features = np.asarray(
        [prepare_video_feature(frame, resolution=192) for frame in frames_bgr],
        dtype=np.uint8,
    )
    source_digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    input_hashes_before = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in required.items()
    }

    temporal_config, optical_flow_enabled, temporal_evidence = (
        load_and_verify_frozen_config(Path(args.temporal_config), Path(args.day65_result))
    )
    occlusion_payload = json.loads(
        Path(args.occlusion_config).read_text(encoding="utf-8")
    )
    if occlusion_payload.get("marker") != "DAY68_FROZEN_CONFIG":
        raise ValueError("occlusion config is not the accepted Day68 freeze")
    occlusion_names = {item.name for item in fields(OcclusionGuardConfig)}
    occlusion_config = OcclusionGuardConfig(
        **{name: occlusion_payload[name] for name in occlusion_names}
    )

    specs = (
        {
            "name": "cpu_eager_bs1",
            "device": "cpu",
            "batch_size": 1,
            "optimized": False,
            "channels_last": False,
        },
        {
            "name": "cpu_optimized_bs1",
            "device": "cpu",
            "batch_size": 1,
            "optimized": True,
            "channels_last": True,
        },
        {
            "name": "cuda_eager_bs1",
            "device": "cuda",
            "batch_size": 1,
            "optimized": False,
            "channels_last": False,
        },
        {
            "name": "cuda_eager_bs32",
            "device": "cuda",
            "batch_size": 32,
            "optimized": False,
            "channels_last": False,
        },
        {
            "name": "cuda_no_tf32_eager_bs1",
            "device": "cuda",
            "batch_size": 1,
            "optimized": False,
            "channels_last": False,
            "cudnn_allow_tf32": False,
        },
        {
            "name": "cuda_no_tf32_eager_bs32",
            "device": "cuda",
            "batch_size": 32,
            "optimized": False,
            "channels_last": False,
            "cudnn_allow_tf32": False,
        },
    )

    probabilities_by_arm: dict[str, np.ndarray] = {}
    observations_by_arm: dict[str, list[list[dict[str, Any]]]] = {}
    final_records_by_arm: dict[str, list[dict[str, Any]]] = {}
    arm_results: dict[str, Any] = {}

    for spec in specs:
        name = str(spec["name"])
        if str(spec["device"]).startswith("cuda"):
            torch.backends.cudnn.allow_tf32 = bool(
                spec.get("cudnn_allow_tf32", True)
            )
        if spec["optimized"]:
            predictor = OptimizedCpuFrozenDay63Predictor(Path(args.checkpoint))
        else:
            predictor = FrozenDay63Predictor(
                Path(args.checkpoint),
                device=str(spec["device"]),
                batch_size=int(spec["batch_size"]),
            )
        model = predictor.model
        repeated: list[np.ndarray] = []
        runtimes: list[float] = []
        run_model_probabilities(
            model,
            features[: min(len(features), int(spec["batch_size"]))],
            device=str(spec["device"]),
            batch_size=int(spec["batch_size"]),
            channels_last=bool(spec["channels_last"]),
        )
        if str(spec["device"]).startswith("cuda"):
            torch.cuda.synchronize()
        for _ in range(args.repeats):
            if str(spec["device"]).startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            probability = run_model_probabilities(
                model,
                features,
                device=str(spec["device"]),
                batch_size=int(spec["batch_size"]),
                channels_last=bool(spec["channels_last"]),
            )
            if str(spec["device"]).startswith("cuda"):
                torch.cuda.synchronize()
            runtimes.append(1000.0 * (time.perf_counter() - started) / len(features))
            repeated.append(probability)

        probability = repeated[0]
        probabilities_by_arm[name] = probability
        instrumented_decoded = [
            decode_centerline_heatmap(
                item,
                peak_height=float(args.peak_height),
                peak_prominence=0.03,
                peak_distance_norm=0.06,
            ).rows
            for item in probability
        ]
        instrumented_serialized = [
            [
                {
                    "near_x_norm": float(row.near_x_norm),
                    "far_x_norm": float(row.far_x_norm),
                    "confidence": float(row.confidence),
                    "support_band_count": int(row.support_band_count),
                }
                for row in frame_rows
            ]
            for frame_rows in instrumented_decoded
        ]
        canonical_decoded = predictor.predict(frames_bgr)
        canonical_serialized = [
            [
                {
                    "near_x_norm": float(row.near_x_norm),
                    "far_x_norm": float(row.far_x_norm),
                    "confidence": float(row.confidence),
                    "support_band_count": int(row.support_band_count),
                }
                for row in frame_rows
            ]
            for frame_rows in canonical_decoded
        ]
        observations_by_arm[name] = canonical_serialized
        canonical_check = compare_observations(
            instrumented_serialized, canonical_serialized
        )

        class PrecomputedPredictor:
            def predict(self, received_frames: list[np.ndarray]) -> list[Any]:
                if len(received_frames) != len(canonical_decoded):
                    raise ValueError("precomputed observations are not frame-aligned")
                return canonical_decoded

        raw_records, _ = process_video_episode(
            video_path,
            predictor=PrecomputedPredictor(),
            config=temporal_config,
            episode=video_path.stem,
            use_optical_flow=optical_flow_enabled,
        )
        projected = [project_navigation_contract(record) for record in raw_records]
        final_records = apply_guard_to_records(
            projected, occlusion_config, frames=frames_bgr
        )
        final_records_by_arm[name] = final_records
        counts = Counter(record["pilot_state"] for record in final_records)
        violations = sum(
            bool(record.get("navigation_available"))
            != (record.get("pilot_state") == "valid")
            for record in final_records
        )
        repeat_comparisons = [
            compare_probability_stacks(repeated[0], item, threshold=args.peak_height)
            for item in repeated[1:]
        ]
        arm_results[name] = {
            "device": spec["device"],
            "backend": (
                "torchscript_optimize_for_inference_channels_last"
                if spec["optimized"]
                else "torch_eager"
            ),
            "batch_size": spec["batch_size"],
            "cudnn_allow_tf32": (
                torch.backends.cudnn.allow_tf32
                if str(spec["device"]).startswith("cuda")
                else None
            ),
            "probability_fingerprint": array_fingerprint(probability),
            "repeat_determinism": repeat_comparisons,
            "runtime_ms_per_frame_repeats": runtimes,
            "runtime_ms_per_frame_median": float(np.median(runtimes)),
            "status_counts": {
                state: int(counts.get(state, 0))
                for state in ("valid", "candidate", "degraded", "reject")
            },
            "navigation_contract_violations": int(violations),
            "instrumented_vs_canonical_predictor": canonical_check,
        }
        del model, predictor, repeated
        if str(spec["device"]).startswith("cuda"):
            torch.cuda.empty_cache()

    pairs = (
        ("cpu_eager_bs1", "cpu_optimized_bs1"),
        ("cpu_eager_bs1", "cuda_eager_bs1"),
        ("cuda_eager_bs1", "cuda_eager_bs32"),
        ("cpu_optimized_bs1", "cuda_eager_bs32"),
        ("cpu_eager_bs1", "cuda_no_tf32_eager_bs1"),
        ("cuda_eager_bs1", "cuda_no_tf32_eager_bs1"),
        ("cpu_eager_bs1", "cuda_no_tf32_eager_bs32"),
        ("cuda_no_tf32_eager_bs1", "cuda_no_tf32_eager_bs32"),
        ("cpu_optimized_bs1", "cuda_no_tf32_eager_bs32"),
        ("cuda_eager_bs32", "cuda_no_tf32_eager_bs32"),
    )
    comparisons: dict[str, Any] = {}
    for left_name, right_name in pairs:
        key = f"{left_name}__{right_name}"
        final = compare_final_records(
            final_records_by_arm[left_name], final_records_by_arm[right_name]
        )
        comparisons[key] = {
            **final,
            "probability": compare_probability_stacks(
                probabilities_by_arm[left_name],
                probabilities_by_arm[right_name],
                threshold=args.peak_height,
            ),
            "decoded_observations": compare_observations(
                observations_by_arm[left_name], observations_by_arm[right_name]
            ),
        }

    root_cause = classify_root_cause(comparisons)
    policy = recommend_execution_policy(comparisons)
    emitted_cli_comparison = None
    cli_provenance = None
    cli_artifacts = (
        Path(args.cpu_cli_jsonl),
        Path(args.stable_cli_jsonl),
        Path(args.cpu_cli_report),
        Path(args.stable_cli_report),
    )
    if all(path.is_file() for path in cli_artifacts):
        emitted_cli_comparison = compare_emitted_jsonl(
            Path(args.cpu_cli_jsonl), Path(args.stable_cli_jsonl)
        )
        cli_provenance = validate_cli_provenance(
            video_path=video_path,
            cpu_jsonl=Path(args.cpu_cli_jsonl),
            cpu_report=Path(args.cpu_cli_report),
            stable_jsonl=Path(args.stable_cli_jsonl),
            stable_report=Path(args.stable_cli_report),
        )
    current_key = "cpu_optimized_bs1__cuda_eager_bs32"
    current_comparison = comparisons[current_key]
    repeated_exact = all(
        comparison["exactly_equal"]
        for arm in arm_results.values()
        for comparison in arm["repeat_determinism"]
    )
    input_hashes_after = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in required.items()
    }
    acceptance = {
        "source_matches_day70_same_byte_video": source_digest
        == EXPECTED_DAY70_VIDEO_SHA256,
        "decoded_exactly_89_frames": len(frames_bgr) == 89 == reported_frames,
        "all_six_arms_completed": len(arm_results) == 6,
        "each_arm_repeat_exact": repeated_exact,
        "day70_37_status_mismatches_reproduced": current_comparison[
            "status_mismatch_frames"
        ]
        == 37,
        "day70_37_navigation_mismatches_reproduced": current_comparison[
            "navigation_mismatch_frames"
        ]
        == 37,
        "all_navigation_contracts_preserved": all(
            arm["navigation_contract_violations"] == 0
            for arm in arm_results.values()
        ),
        "frozen_inputs_unchanged_during_run": input_hashes_before
        == input_hashes_after,
        "diagnostic_factor_found": root_cause["status"] != "UNRESOLVED",
        "stable_cuda_matches_production_cpu_all_89_frames": comparisons[
            "cpu_optimized_bs1__cuda_no_tf32_eager_bs32"
        ]["status_mismatch_frames"]
        == 0
        and comparisons[
            "cpu_optimized_bs1__cuda_no_tf32_eager_bs32"
        ]["navigation_mismatch_frames"]
        == 0,
        "instrumentation_matches_canonical_predictors": all(
            arm["instrumented_vs_canonical_predictor"][
                "row_count_mismatch_frames"
            ]
            == 0
            and arm["instrumented_vs_canonical_predictor"][
                "max_matched_near_x_abs_difference"
            ]
            == 0
            and arm["instrumented_vs_canonical_predictor"][
                "max_matched_far_x_abs_difference"
            ]
            == 0
            for arm in arm_results.values()
        ),
        "emitted_cpu_and_stable_cuda_jsonl_match_framewise": (
            emitted_cli_comparison is not None
            and emitted_cli_comparison["status_mismatch_frames"] == 0
            and emitted_cli_comparison["navigation_mismatch_frames"] == 0
            and emitted_cli_comparison["frame_identity_mismatch_frames"] == 0
            and emitted_cli_comparison["geometry_mismatch_frames"] == 0
        ),
        "emitted_cli_artifacts_bound_to_validated_reports": (
            cli_provenance is not None and cli_provenance["valid"]
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.differences_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.differences_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = ["frame_index"]
        for spec in specs:
            fieldnames.extend(
                [f"{spec['name']}_state", f"{spec['name']}_row_count"]
            )
        for left_name, right_name in pairs:
            fieldnames.append(f"{left_name}__{right_name}_max_probability_abs_diff")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in range(len(frames_bgr)):
            row: dict[str, Any] = {"frame_index": frame_index}
            for spec in specs:
                name = str(spec["name"])
                row[f"{name}_state"] = final_records_by_arm[name][frame_index][
                    "pilot_state"
                ]
                row[f"{name}_row_count"] = len(
                    observations_by_arm[name][frame_index]
                )
            for left_name, right_name in pairs:
                row[f"{left_name}__{right_name}_max_probability_abs_diff"] = float(
                    np.max(
                        np.abs(
                            probabilities_by_arm[left_name][frame_index].astype(np.float64)
                            - probabilities_by_arm[right_name][frame_index].astype(np.float64)
                        )
                    )
                )
            writer.writerow(row)

    result = {
        "schema_version": 1,
        "marker": "DAY71_DEVICE_PARITY_DIAGNOSIS_COMPLETE",
        "date": "2026-09-12",
        "evidence_role": "DEVELOPMENT_REPRODUCIBILITY_DIAGNOSIS_NOT_MODEL_SELECTION",
        "claim_boundary": (
            "No model, threshold, frozen external result, safety claim, or robot-control claim was changed."
        ),
        "source": {
            "video": str(video_path),
            "sha256": source_digest,
            "reported_frames": reported_frames,
            "decoded_frames": len(frames_bgr),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "feature_fingerprint": array_fingerprint(features),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "frozen_inputs_sha256": input_hashes_after,
        "temporal_freeze_evidence": temporal_evidence,
        "arms": arm_results,
        "comparisons": comparisons,
        "root_cause": root_cause,
        "recommended_execution_policy": policy,
        "emitted_cli_comparison": emitted_cli_comparison,
        "emitted_cli_provenance": cli_provenance,
        "acceptance_checks": acceptance,
        "acceptance_status": "PASS" if all(acceptance.values()) else "FAILED",
        "differences_csv": str(Path(args.differences_csv).resolve()),
    }
    Path(args.report).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["report"] = str(Path(args.report).resolve())
    return result


def main() -> int:
    result = run_diagnosis(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(result["marker"])
    return 0 if result["acceptance_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
