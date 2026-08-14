import csv
import json
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18_baseline"

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CLASS_MAP_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "class_to_idx.json"
)
REPORT_PATH = OUTPUT_DIR / "per_class_metrics.csv"


def main():
    class_to_idx = json.loads(
        CLASS_MAP_PATH.read_text(encoding="utf-8")
    )
    idx_to_class = {
        index: name for name, index in class_to_idx.items()
    }

    with PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        predictions = list(csv.DictReader(file))

    y_true = [int(row["true_index"]) for row in predictions]
    y_pred = [
        int(row["predicted_index"]) for row in predictions
    ]
    labels = list(range(50))

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    report_rows = []

    for class_index in labels:
        correct = int(matrix[class_index, class_index])
        class_support = int(support[class_index])

        report_rows.append(
            {
                "class_index": class_index,
                "class_name": idx_to_class[class_index],
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1_score": float(f1[class_index]),
                "support": class_support,
                "correct": correct,
                "incorrect": class_support - correct,
            }
        )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_index",
                "class_name",
                "precision",
                "recall",
                "f1_score",
                "support",
                "correct",
                "incorrect",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    weakest_classes = sorted(
        report_rows,
        key=lambda row: (
            row["f1_score"],
            row["recall"],
            row["class_index"],
        ),
    )[:10]

    calculated_macro_f1 = sum(
        row["f1_score"] for row in report_rows
    ) / len(report_rows)

    saved_metrics = json.loads(
        TEST_METRICS_PATH.read_text(encoding="utf-8")
    )

    print("类别数量:", len(report_rows))
    print("测试图片:", sum(row["support"] for row in report_rows))
    print("逐类别Macro-F1:", round(calculated_macro_f1, 4))
    print("\nF1最低的10个类别：")

    for row in weakest_classes:
        print(
            f"{row['class_index']:02d} "
            f"{row['class_name']} | "
            f"P={row['precision']:.3f} | "
            f"R={row['recall']:.3f} | "
            f"F1={row['f1_score']:.3f} | "
            f"样本={row['support']} | "
            f"错误={row['incorrect']}"
        )

    print("\n逐类别报告:", REPORT_PATH)

    assert len(predictions) == 823
    assert len(report_rows) == 50
    assert sum(row["support"] for row in report_rows) == 823
    assert abs(
        calculated_macro_f1
        - saved_metrics["test_macro_f1"]
    ) < 1e-10
    assert REPORT_PATH.is_file()

    print("第五天第二步验证成功")


if __name__ == "__main__":
    main()