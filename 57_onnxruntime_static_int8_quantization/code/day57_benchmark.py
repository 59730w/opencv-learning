from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def speedup(fp32_median_ms: float, int8_median_ms: float) -> float:
    if fp32_median_ms <= 0 or int8_median_ms <= 0:
        raise ValueError("latencies must be positive")
    return fp32_median_ms / int8_median_ms


def run_checked(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0 or "DAY55_BENCHMARK_OK" not in result.stdout:
        raise RuntimeError(f"benchmark failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="复用 Day55 C++ 程序公平比较 FP32 与 INT8")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs=6, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, model_path in (("fp32", args.fp32), ("int8", args.int8)):
        for batch in (1, 6):
            prefix = f"{model_name}_batch{batch}"
            raw = args.output_dir / f"{prefix}_raw.csv"
            summary = args.output_dir / f"{prefix}_summary.csv"
            logits = args.output_dir / f"{prefix}_logits.bin"
            run_checked([
                str(args.executable), "--model", str(model_path), "--class-map", str(args.class_map),
                "--mode", "runtime-only", "--batch-size", str(batch), "--warmup", "3",
                "--runs", "10", "--intra-threads", "0", "--raw-output", str(raw),
                "--summary-output", str(summary), "--logits-output", str(logits),
                "--images", *[str(path) for path in args.images],
            ])
            with summary.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            rows.append({"model": model_name, "model_size_bytes": model_path.stat().st_size, **row})
    by_batch = {(row["model"], int(row["batch_size"])): row for row in rows}
    for row in rows:
        batch = int(row["batch_size"])
        row["speedup_vs_fp32"] = (
            f"{speedup(float(by_batch[('fp32', batch)]['median_ms']), float(row['median_ms'])):.6f}"
            if row["model"] == "int8" else "1.000000"
        )
    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.combined_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']} batch={row['batch_size']} median={row['median_ms']} ms "
              f"images/s={row['images_per_second']} speedup={row['speedup_vs_fp32']}x")
    print("DAY57_BENCHMARK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
