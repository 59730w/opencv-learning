import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def summarize_events(events: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in events:
        args = event.get("args") or {}
        if event.get("cat") != "Node" or not args.get("op_name") or not args.get("provider"):
            continue
        duration = event.get("dur")
        if not isinstance(duration, (int, float)) or duration < 0:
            continue
        grouped[(str(args["op_name"]), str(args["provider"]))].append(float(duration))
    if not grouped:
        raise ValueError("profile contains no usable operator events")
    grand_total = sum(sum(values) for values in grouped.values())
    rows = []
    for (op_name, provider), values in grouped.items():
        total = sum(values)
        rows.append({
            "op_name": op_name,
            "provider": provider,
            "event_count": len(values),
            "total_duration_us": total,
            "mean_duration_us": total / len(values),
            "share_percent": 100.0 * total / grand_total if grand_total else 0.0,
        })
    return sorted(rows, key=lambda row: (-row["total_duration_us"], row["op_name"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ONNX Runtime profiling 中的算子事件")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")
    events = json.loads(args.profile.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("profile root must be a JSON array")
    rows = summarize_events(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["op_name", "provider", "event_count", "total_duration_us",
              "mean_duration_us", "share_percent"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row,
                             "total_duration_us": f"{row['total_duration_us']:.3f}",
                             "mean_duration_us": f"{row['mean_duration_us']:.3f}",
                             "share_percent": f"{row['share_percent']:.3f}"})
    print(f"Operator groups: {len(rows)}")
    print("Top operators:")
    for row in rows[:args.top]:
        print(f"  {row['op_name']:<24} {row['share_percent']:7.3f}%  "
              f"total={row['total_duration_us']:.3f} us  count={row['event_count']}")
    print("DAY56_PROFILE_ANALYSIS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
