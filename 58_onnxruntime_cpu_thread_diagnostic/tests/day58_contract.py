import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the tracked Day58 contract and measured outputs")
    parser.add_argument("--lesson-root", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--recorded-result", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads((args.lesson_root / "thread_experiment_contract.json").read_text(encoding="utf-8"))
    if contract["independent_variable"]["values"] != [0, 1]:
        raise AssertionError("thread values changed")
    fixed = contract["fixed_variables"]
    if fixed["warmup_runs"] != 5 or fixed["measured_runs"] != 30:
        raise AssertionError("measurement counts changed")
    if fixed["batch_sizes"] != [1, 6] or contract["matrix_cell_count"] != 8:
        raise AssertionError("matrix changed")

    with args.combined_output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["model"], int(row["batch_size"]), int(row["intra_threads"])) for row in rows}
    expected = {(model, batch, threads) for model in ("fp32", "int8")
                for batch in (1, 6) for threads in (0, 1)}
    if len(rows) != 8 or keys != expected:
        raise AssertionError("measured matrix is incomplete")
    if any(int(row["raw_run_count"]) != 30 for row in rows):
        raise AssertionError("raw measurement count changed")

    evidence = json.loads(args.evidence_output.read_text(encoding="utf-8"))
    if evidence["matrix_cell_count"] != 8 or evidence["raw_timing_count"] != 240:
        raise AssertionError("evidence counts are incorrect")
    if len(evidence["correctness_checks"]) != 4:
        raise AssertionError("correctness matrix is incomplete")
    if not all(item["ordered_top3_equal"] for item in evidence["correctness_checks"]):
        raise AssertionError("correctness gate failed")

    recorded = json.loads(args.recorded_result.read_text(encoding="utf-8"))
    if recorded["matrix_cell_count"] != 8 or recorded["raw_timing_count"] != 240:
        raise AssertionError("recorded result counts are incorrect")
    if len(recorded["results"]) != 8:
        raise AssertionError("recorded result matrix is incomplete")
    if evidence["interpretation"] != recorded["interpretation"]:
        raise AssertionError("rerun interpretation differs from the recorded result")
    for batch in ("1", "6"):
        rerun = evidence["by_batch"][batch]
        if rerun["default_int8_speedup_vs_fp32"] >= 1.0:
            raise AssertionError(f"batch {batch} no longer reproduces the default-thread slowdown")
        if rerun["single_thread_int8_speedup_vs_fp32"] <= 1.0:
            raise AssertionError(f"batch {batch} no longer reproduces the single-thread reversal")

    notes = (args.lesson_root / "code" / "day58_notes.md").read_text(encoding="utf-8")
    for row in recorded["results"]:
        for field in ("median_ms", "p90_ms"):
            rendered = str(Decimal(row[field]).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
            if rendered not in notes:
                raise AssertionError(f"notes do not reconcile {field}={rendered}")
    interpretation = recorded["interpretation"]
    if f"Interpretation: {interpretation}" not in notes:
        raise AssertionError("notes do not reconcile the interpretation")
    for marker in ("DAY58_CORRECTNESS_OK", "DAY58_THREAD_EXPERIMENT_OK", "DAY58_CONTRACT_OK"):
        if marker not in notes:
            raise AssertionError(f"notes omit success marker: {marker}")

    print("DAY58_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
