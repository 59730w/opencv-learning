import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"expected exit {expected}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--day54-executable", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--images", nargs=6, required=True)
    args = parser.parse_args()

    executable = Path(args.executable)
    if not executable.is_file():
        raise AssertionError(f"benchmark executable missing: {executable}")

    with tempfile.TemporaryDirectory(prefix="day55_contract_") as temp_dir:
        temp = Path(temp_dir)
        raw_csv = temp / "raw.csv"
        summary_csv = temp / "summary.csv"
        logits = temp / "day55_logits.bin"
        day54_tensor = temp / "day54_tensor.bin"
        day54_logits = temp / "day54_logits.bin"

        common = [
            str(executable), "--model", args.model, "--class-map", args.class_map,
            "--mode", "runtime-only", "--batch-size", "3", "--warmup", "1",
            "--runs", "2", "--intra-threads", "1", "--raw-output", str(raw_csv),
            "--summary-output", str(summary_csv), "--logits-output", str(logits),
            "--images", *args.images,
        ]
        result = run(common)
        for marker in (
            "Mode: runtime-only", "Batch size: 3", "Warm-up runs: 1",
            "Measured runs: 2", "Intra-op threads: 1", "DAY55_BENCHMARK_OK",
            "Timed boundary: Session::Run only", "Input tensor creations: 1",
        ):
            if marker not in result.stdout:
                raise AssertionError(f"missing marker: {marker}")

        with raw_csv.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        if len(raw_rows) != 2 or {row["run_index"] for row in raw_rows} != {"1", "2"}:
            raise AssertionError("raw CSV rows are incorrect")

        with summary_csv.open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        if len(summary_rows) != 1 or summary_rows[0]["batch_size"] != "3":
            raise AssertionError("summary CSV row is incorrect")
        if float(summary_rows[0]["session_creation_ms"]) <= 0.0:
            raise AssertionError("session creation time was not retained")

        run([
            args.day54_executable, "--model", args.model, "--class-map", args.class_map,
            "--tensor-output", str(day54_tensor), "--logits-output", str(day54_logits),
            "--images", *args.images[:3],
        ])
        reference = np.fromfile(day54_logits, dtype=np.float32)
        candidate = np.fromfile(logits, dtype=np.float32)
        if reference.shape != candidate.shape or reference.shape != (150,):
            raise AssertionError("logit shape mismatch")
        if float(np.max(np.abs(reference - candidate))) > 1e-6:
            raise AssertionError("Day54/Day55 logits differ")
        if not np.array_equal(
            np.argsort(-reference.reshape(3, 50), axis=1)[:, :3],
            np.argsort(-candidate.reshape(3, 50), axis=1)[:, :3],
        ):
            raise AssertionError("ordered Top-3 differs")

        end_to_end = common.copy()
        end_to_end[end_to_end.index("runtime-only")] = "end-to-end"
        result = run(end_to_end)
        for marker in (
            "Mode: end-to-end",
            "Timed boundary: decode + preprocess + tensor + Session::Run + Top-k",
            "Input tensor creations: 3",
        ):
            if marker not in result.stdout:
                raise AssertionError(f"missing end-to-end marker: {marker}")

        invalid = common.copy()
        invalid[invalid.index("3", invalid.index("--batch-size"))] = "2"
        run(invalid, expected=2)

    print("DAY55_BENCHMARK_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
