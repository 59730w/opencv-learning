"""Day71 device-stable entry point for the frozen Day69 crop-row pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY69_CODE = PROJECT_ROOT / "69_crop_row_frozen_evaluation" / "code"


def configure_device_stable_execution() -> dict[str, object]:
    """Apply the Day71 execution-only policy validated on the audit video."""
    torch.backends.cudnn.allow_tf32 = False
    return {
        "marker": "DAY71_DEVICE_STABLE_POLICY_APPLIED",
        "policy": "cuda_no_tf32_eager_batch32",
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "effective_device": "cuda",
        "effective_batch_size": 32,
        "changes_model_or_thresholds": False,
        "evidence_boundary": "validated on one 89-frame shifted-development video",
    }


def with_validated_execution(arguments: Sequence[str]) -> list[str]:
    """Require the exact CUDA/batch policy supported by Day71 evidence."""
    result = list(arguments)
    device = None
    batch = None
    for index, value in enumerate(result):
        if value == "--device" and index + 1 < len(result):
            device = result[index + 1]
        elif value.startswith("--device="):
            device = value.split("=", 1)[1]
        elif value == "--batch-size" and index + 1 < len(result):
            batch = result[index + 1]
        elif value.startswith("--batch-size="):
            batch = value.split("=", 1)[1]
    if device is None:
        result.extend(["--device", "cuda"])
    elif device.lower() != "cuda":
        raise ValueError("Day71 stable runner requires --device cuda")
    if batch is None:
        result.extend(["--batch-size", "32"])
    elif batch != "32":
        raise ValueError("Day71 stable runner requires --batch-size 32")
    return result


def persist_policy(report_path: Path, policy: dict[str, object]) -> None:
    """Attach the effective Day71 execution contract to the canonical report."""
    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["day71_execution_policy"] = policy
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    policy = configure_device_stable_execution()
    sys.argv[1:] = with_validated_execution(sys.argv[1:])
    output_dir = None
    for index, value in enumerate(sys.argv[1:]):
        if value == "--output-dir" and index + 2 < len(sys.argv):
            output_dir = Path(sys.argv[index + 2])
        elif value.startswith("--output-dir="):
            output_dir = Path(value.split("=", 1)[1])
    if str(DAY69_CODE) not in sys.path:
        sys.path.insert(0, str(DAY69_CODE))
    from run_crop_row_pilot import main as run_frozen_pilot

    print(json.dumps(policy, ensure_ascii=False))
    status = run_frozen_pilot()
    if status == 0 and output_dir is not None:
        persist_policy(output_dir / "pilot_report.json", policy)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
