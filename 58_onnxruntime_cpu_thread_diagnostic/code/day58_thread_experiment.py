from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


WARMUP_RUNS = 5
MEASURED_RUNS = 30


def experiment_matrix() -> list[dict[str, int | str]]:
    schedule = (
        (1, 0, ("fp32", "int8")),
        (1, 1, ("int8", "fp32")),
        (6, 0, ("int8", "fp32")),
        (6, 1, ("fp32", "int8")),
    )
    return [
        {"model": model, "batch_size": batch, "intra_threads": threads}
        for batch, threads, model_order in schedule
        for model in model_order
    ]


def ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("ratio values must be positive")
    return numerator / denominator


def build_command(
    executable: Path,
    model: Path,
    class_map: Path,
    images: list[Path],
    batch_size: int,
    intra_threads: int,
    raw_output: Path,
    summary_output: Path,
    logits_output: Path,
) -> list[str]:
    if len(images) != 6:
        raise ValueError("exactly six benchmark images are required")
    return [
        str(executable),
        "--model", str(model),
        "--class-map", str(class_map),
        "--mode", "runtime-only",
        "--batch-size", str(batch_size),
        "--warmup", str(WARMUP_RUNS),
        "--runs", str(MEASURED_RUNS),
        "--intra-threads", str(intra_threads),
        "--raw-output", str(raw_output),
        "--summary-output", str(summary_output),
        "--logits-output", str(logits_output),
        "--images", *[str(path) for path in images],
    ]


def run_checked(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0 or "DAY55_BENCHMARK_OK" not in result.stdout:
        raise RuntimeError(
            "benchmark failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def validate_same_model_logits(
    default_path: Path, single_path: Path, batch_size: int
) -> dict[str, int | float | bool]:
    default = np.fromfile(default_path, dtype=np.float32)
    single = np.fromfile(single_path, dtype=np.float32)
    expected_size = batch_size * 50
    if default.size != expected_size or single.size != expected_size:
        raise ValueError("unexpected logit shape")
    if not np.all(np.isfinite(default)) or not np.all(np.isfinite(single)):
        raise ValueError("non-finite logits")
    default = default.reshape(batch_size, 50)
    single = single.reshape(batch_size, 50)
    default_top3 = np.argsort(-default, axis=1)[:, :3]
    single_top3 = np.argsort(-single, axis=1)[:, :3]
    ordered_equal = bool(np.array_equal(default_top3, single_top3))
    if not ordered_equal:
        raise ValueError("ordered Top-3 differs across thread settings")
    absolute = np.abs(default - single)
    return {
        "batch_size": batch_size,
        "logit_count": int(default.size),
        "ordered_top3_equal": ordered_equal,
        "mean_absolute_difference": float(np.mean(absolute)),
        "maximum_absolute_difference": float(np.max(absolute)),
    }


def enrich_rows(
    rows: list[dict[str, str]], practical_margin: float = 0.02
) -> tuple[list[dict[str, str]], dict[str, object]]:
    copied = [dict(row) for row in rows]
    by_key: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in copied:
        key = (row["model"], int(row["batch_size"]), int(row["intra_threads"]))
        if key in by_key:
            raise ValueError(f"duplicate experiment cell: {key}")
        if float(row["median_ms"]) <= 0.0:
            raise ValueError(f"non-positive median: {key}")
        by_key[key] = row
    expected = {
        (model, batch, threads)
        for model in ("fp32", "int8")
        for batch in (1, 6)
        for threads in (0, 1)
    }
    if set(by_key) != expected:
        raise ValueError("experiment matrix is incomplete")

    for key, row in by_key.items():
        model, batch, threads = key
        fp32_median = float(by_key[("fp32", batch, threads)]["median_ms"])
        model_median = float(row["median_ms"])
        default_median = float(by_key[(model, batch, 0)]["median_ms"])
        row["speedup_vs_fp32_same_threads"] = f"{ratio(fp32_median, model_median):.9f}"
        row["default_vs_single_thread_ratio"] = (
            f"{ratio(default_median, model_median):.9f}" if threads == 1 else "1.000000000"
        )

    batch_evidence: dict[str, dict[str, float]] = {}
    improvements = []
    single_speedups = []
    for batch in (1, 6):
        default_speedup = ratio(
            float(by_key[("fp32", batch, 0)]["median_ms"]),
            float(by_key[("int8", batch, 0)]["median_ms"]),
        )
        single_speedup = ratio(
            float(by_key[("fp32", batch, 1)]["median_ms"]),
            float(by_key[("int8", batch, 1)]["median_ms"]),
        )
        relative_improvement = ratio(single_speedup, default_speedup) - 1.0
        improvements.append(relative_improvement)
        single_speedups.append(single_speedup)
        batch_evidence[str(batch)] = {
            "default_int8_speedup_vs_fp32": default_speedup,
            "single_thread_int8_speedup_vs_fp32": single_speedup,
            "relative_speedup_improvement": relative_improvement,
        }
    if all(value >= 1.0 for value in single_speedups) and all(
        value >= practical_margin for value in improvements
    ):
        interpretation = "full_support"
    elif any(value >= practical_margin for value in improvements):
        interpretation = "partial_support"
    else:
        interpretation = "not_supported"
    evidence: dict[str, object] = {
        "practical_improvement_margin": practical_margin,
        "by_batch": batch_evidence,
        "interpretation": interpretation,
    }
    return copied, evidence


def read_single_summary(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one summary row: {path}")
    return rows[0]


def count_raw_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != MEASURED_RUNS:
        raise ValueError(f"expected {MEASURED_RUNS} raw rows, got {count}: {path}")
    return count


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled Day58 ONNX Runtime intra-op thread diagnostic"
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs=6, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()

    for path, label in (
        (args.executable, "benchmark executable"),
        (args.fp32, "FP32 model"),
        (args.int8, "INT8 model"),
        (args.class_map, "class map"),
    ):
        require_file(path, label)
    for image in args.images:
        require_file(image, "benchmark image")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {"fp32": args.fp32, "int8": args.int8}
    rows: list[dict[str, str]] = []
    logits_paths: dict[tuple[str, int, int], Path] = {}
    raw_timing_count = 0
    for cell in experiment_matrix():
        model_name = str(cell["model"])
        batch = int(cell["batch_size"])
        threads = int(cell["intra_threads"])
        prefix = f"{model_name}_batch{batch}_intra{threads}"
        raw = args.output_dir / f"{prefix}_raw.csv"
        summary = args.output_dir / f"{prefix}_summary.csv"
        logits = args.output_dir / f"{prefix}_logits.bin"
        run_checked(build_command(
            args.executable, model_paths[model_name], args.class_map, list(args.images),
            batch, threads, raw, summary, logits,
        ))
        raw_count = count_raw_rows(raw)
        raw_timing_count += raw_count
        row = read_single_summary(summary)
        if int(row["batch_size"]) != batch or int(row["intra_threads"]) != threads:
            raise ValueError(f"summary configuration mismatch: {summary}")
        rows.append({
            "model": model_name,
            "model_size_bytes": str(model_paths[model_name].stat().st_size),
            **row,
            "raw_run_count": str(raw_count),
        })
        logits_paths[(model_name, batch, threads)] = logits

    correctness_checks = []
    for model_name in ("fp32", "int8"):
        for batch in (1, 6):
            check = validate_same_model_logits(
                logits_paths[(model_name, batch, 0)],
                logits_paths[(model_name, batch, 1)],
                batch,
            )
            correctness_checks.append({"model": model_name, **check})
    print("DAY58_CORRECTNESS_OK")

    enriched, evidence = enrich_rows(rows)
    evidence.update({
        "matrix_cell_count": len(enriched),
        "raw_timing_count": raw_timing_count,
        "correctness_checks": correctness_checks,
    })
    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    with args.combined_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for row in enriched:
        print(
            f"{row['model']} batch={row['batch_size']} intra={row['intra_threads']} "
            f"median={row['median_ms']} ms p90={row['p90_ms']} ms "
            f"speedup={row['speedup_vs_fp32_same_threads']}x"
        )
    print(f"Interpretation: {evidence['interpretation']}")
    print("DAY58_THREAD_EXPERIMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
