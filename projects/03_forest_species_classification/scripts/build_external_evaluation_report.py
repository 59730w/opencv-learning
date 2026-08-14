import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_analysis import (
    find_adjacent_cross_split_pairs,
    render_external_report,
)


OFFICIAL_METRICS_PATH = PROJECT_ROOT / "outputs" / "resnet18_baseline" / "test_metrics.json"
STRICT_METRICS_PATH = PROJECT_ROOT / "outputs" / "external_test" / "external_metrics.json"
STRICT_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "external_test" / "external_predictions.csv"
POSTHOC_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "external_test_posthoc_multicrop"
    / "multicrop_metrics.json"
)
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "split_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_test"
SEQUENCE_RISK_PATH = OUTPUT_DIR / "sequence_split_risk.json"
GAP_CHART_PATH = OUTPUT_DIR / "generalization_gap.png"
BIAS_CHART_PATH = OUTPUT_DIR / "external_prediction_bias.png"
REPORT_PATH = OUTPUT_DIR / "external_evaluation_report.md"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_generalization_gap_chart(official, strict):
    labels = ["BarkVN-50\ntest Top-1", "Strict external\nTop-1", "Strict external\nTop-3"]
    values = [
        official["test_accuracy"] * 100,
        strict["positive"]["top1_accuracy"] * 100,
        strict["positive"]["top3_accuracy"] * 100,
    ]
    colors = ["#2f7ed8", "#d9534f", "#f0ad4e"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("In-domain and strict external evaluation")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(GAP_CHART_PATH, dpi=180)
    plt.close(fig)


def save_prediction_bias_chart(prediction_counts):
    ordered = sorted(
        prediction_counts.items(),
        key=lambda item: (item[1], item[0]),
    )[-10:]
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color="#d9534f")
    ax.set_xlabel("Top-1 prediction count among 20 positives")
    ax.set_title("Strict external prediction bias")
    ax.set_xlim(0, max(values + [1]) + 1)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.08,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
        )
    fig.tight_layout()
    fig.savefig(BIAS_CHART_PATH, dpi=180)
    plt.close(fig)


def main():
    official = load_json(OFFICIAL_METRICS_PATH)
    strict = load_json(STRICT_METRICS_PATH)
    posthoc = load_json(POSTHOC_METRICS_PATH)
    strict_predictions = load_csv(STRICT_PREDICTIONS_PATH)
    manifest_rows = load_csv(MANIFEST_PATH)

    positive_predictions = [
        row for row in strict_predictions if row["test_type"] == "positive"
    ]
    prediction_counts = Counter(
        row["top1_class"] for row in positive_predictions
    )
    sequence_pairs = find_adjacent_cross_split_pairs(
        manifest_rows,
        max_gap=2,
    )
    classes_involved = len({pair["class_name"] for pair in sequence_pairs})

    assert official["test_images"] == 823
    assert abs(official["test_accuracy"] - 0.9805589307411907) < 1e-12
    assert strict["positive"]["images"] == 20
    assert strict["positive"]["top1_correct"] == 1
    assert strict["positive"]["top3_correct"] == 4
    assert strict["negative"]["images"] == 6
    assert posthoc["strict_external_test"] is False
    assert posthoc["external_multicrop"]["positive"]["top1_correct"] == 2
    assert posthoc["external_multicrop"]["positive"]["top3_correct"] == 3
    assert len(sequence_pairs) == 2442
    assert classes_involved == 50

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sequence_evidence = {
        "heuristic": (
            "Within each class, sort numeric image filenames and count adjacent pairs "
            "whose number gap is at most 2 and whose split differs."
        ),
        "interpretation": (
            "Risk indicator for acquisition-sequence leakage; not proof that every pair "
            "shows the same physical tree."
        ),
        "max_number_gap": 2,
        "manifest_images": len(manifest_rows),
        "adjacent_cross_split_pairs": len(sequence_pairs),
        "classes_involved": classes_involved,
        "pairs": sequence_pairs,
    }
    SEQUENCE_RISK_PATH.write_text(
        json.dumps(sequence_evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_generalization_gap_chart(official, strict)
    save_prediction_bias_chart(prediction_counts)

    posthoc_positive = posthoc["external_multicrop"]["positive"]
    evidence = {
        "official_test": {
            "images": official["test_images"],
            "accuracy": official["test_accuracy"],
            "macro_f1": official["test_macro_f1"],
        },
        "strict": {
            "positive_images": strict["positive"]["images"],
            "top1_correct": strict["positive"]["top1_correct"],
            "top1_accuracy": strict["positive"]["top1_accuracy"],
            "top3_correct": strict["positive"]["top3_correct"],
            "top3_accuracy": strict["positive"]["top3_accuracy"],
        },
        "posthoc": {
            "strict_external_test": posthoc["strict_external_test"],
            "top1_correct": posthoc_positive["top1_correct"],
            "top1_accuracy": posthoc_positive["top1_accuracy"],
            "top3_correct": posthoc_positive["top3_correct"],
            "top3_accuracy": posthoc_positive["top3_accuracy"],
        },
        "sequence_risk": {
            "adjacent_cross_split_pairs": len(sequence_pairs),
            "classes_involved": classes_involved,
        },
        "view_metrics": strict["positive"]["by_view"],
        "class_metrics": strict["positive"]["by_class"],
        "negative": strict["negative"],
        "prediction_counts": dict(prediction_counts),
    }
    REPORT_PATH.write_text(
        render_external_report(evidence),
        encoding="utf-8",
    )

    print("同类别相邻编号跨子集对:", len(sequence_pairs))
    print("涉及类别:", classes_involved)
    print("序列风险证据:", SEQUENCE_RISK_PATH)
    print("泛化落差图:", GAP_CHART_PATH)
    print("预测偏向图:", BIAS_CHART_PATH)
    print("外部评估报告:", REPORT_PATH)
    print("第六天第五步报告生成成功；未训练、调参或修改外部图片")


if __name__ == "__main__":
    main()
