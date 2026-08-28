import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"expected exit {expected}, got {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--analyzer", required=True)
    parser.add_argument("--day54-executable", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--images", nargs=6, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="day56_contract_") as temp_dir:
        temp = Path(temp_dir)
        profile, summary = temp / "profile.json", temp / "summary.csv"
        logits = temp / "day56_logits.bin"
        reference_tensor, reference_logits = temp / "day54_tensor.bin", temp / "day54_logits.bin"
        command = [
            args.executable, "--model", args.model, "--batch-size", "1",
            "--warmup", "1", "--runs", "2", "--intra-threads", "1",
            "--profile-output", str(profile), "--logits-output", str(logits),
            "--images", *args.images,
        ]
        result = run(command)
        for marker in ("Batch size: 1", "Warm-up runs (separate session): 1",
                       "Profiled runs: 2", "Intra-op threads: 1", "DAY56_PROFILING_OK"):
            if marker not in result.stdout:
                raise AssertionError(f"missing marker: {marker}")
        if not profile.is_file() or not logits.is_file():
            raise AssertionError("profiling outputs missing")
        events = json.loads(profile.read_text(encoding="utf-8"))
        if not isinstance(events, list) or not any(event.get("cat") == "Node" for event in events):
            raise AssertionError("profile has no Node events")

        analysis = run([sys.executable, args.analyzer, "--profile", str(profile),
                        "--output", str(summary), "--top", "5"])
        if "DAY56_PROFILE_ANALYSIS_OK" not in analysis.stdout:
            raise AssertionError("analysis marker missing")
        with summary.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or float(rows[0]["total_duration_us"]) <= 0.0:
            raise AssertionError("operator summary is empty")

        run([args.day54_executable, "--model", args.model, "--class-map", args.class_map,
             "--tensor-output", str(reference_tensor), "--logits-output", str(reference_logits),
             "--images", args.images[0]])
        reference = np.fromfile(reference_logits, dtype=np.float32)
        candidate = np.fromfile(logits, dtype=np.float32)
        if reference.shape != candidate.shape or reference.shape != (50,):
            raise AssertionError("logit shape mismatch")
        if float(np.max(np.abs(reference - candidate))) > 1e-6:
            raise AssertionError("Day54/Day56 logits differ")
        if not np.array_equal(np.argsort(-reference)[:3], np.argsort(-candidate)[:3]):
            raise AssertionError("ordered Top-3 differs")

        invalid = command.copy()
        invalid[invalid.index("1", invalid.index("--batch-size"))] = "3"
        run(invalid, expected=2)

    print("DAY56_PROFILING_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
