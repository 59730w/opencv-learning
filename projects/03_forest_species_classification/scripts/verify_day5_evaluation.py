import csv
import json
from pathlib import Path

import cv2
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18_baseline"

METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
CLASS_REPORT_PATH = OUTPUT_DIR / "per_class_metrics.csv"
CONFUSION_PAIRS_PATH = OUTPUT_DIR / "confusion_pairs.csv"
ERROR_ANALYSIS_PATH = PROJECT_ROOT / "docs" / "test_error_analysis.md"

IMAGE_PATHS = [
    OUTPUT_DIR / "training_curves.png",
    OUTPUT_DIR / "confusion_matrix_normalized.png",
    OUTPUT_DIR / "confusion_pairs.png",
    OUTPUT_DIR / "misclassified_contact_sheet.jpg",
]


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main():
    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )
    predictions = read_csv(PREDICTIONS_PATH)
    class_report = read_csv(CLASS_REPORT_PATH)
    confusion_pairs = read_csv(CONFUSION_PAIRS_PATH)

    y_true = [int(row["true_index"]) for row in predictions]
    y_pred = [
        int(row["predicted_index"]) for row in predictions
    ]

    correct_count = sum(
        true_index == predicted_index
        for true_index, predicted_index
        in zip(y_true, y_pred)
    )
    error_count = len(predictions) - correct_count

    calculated_accuracy = accuracy_score(y_true, y_pred)
    calculated_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(50)),
        average="macro",
        zero_division=0,
    )

    class_support = sum(
        int(row["support"]) for row in class_report
    )
    confusion_error_count = sum(
        int(row["count"]) for row in confusion_pairs
    )

    unique_paths = {
        row["relative_path"] for row in predictions
    }

    readable_images = []

    for image_path in IMAGE_PATHS:
        image = cv2.imread(str(image_path))
        readable_images.append(image is not None)
        print(
            f"{image_path.name}:",
            "可读取" if image is not None else "无法读取",
        )

    document = ERROR_ANALYSIS_PATH.read_text(
        encoding="utf-8"
    )

    print("\n预测记录:", len(predictions))
    print("唯一路径:", len(unique_paths))
    print("正确预测:", correct_count)
    print("错误预测:", error_count)
    print("重新计算Accuracy:", f"{calculated_accuracy:.6f}")
    print("重新计算Macro-F1:", f"{calculated_macro_f1:.6f}")
    print("逐类别数量:", len(class_report))
    print("逐类别支持数合计:", class_support)
    print("错误类别对数量合计:", confusion_error_count)

    assert metrics["checkpoint_epoch"] == 13
    assert len(predictions) == 823
    assert len(unique_paths) == 823
    assert correct_count == 807
    assert error_count == 16

    assert abs(
        calculated_accuracy - metrics["test_accuracy"]
    ) < 1e-12
    assert abs(
        calculated_macro_f1 - metrics["test_macro_f1"]
    ) < 1e-12

    assert len(class_report) == 50
    assert class_support == 823
    assert confusion_error_count == 16
    assert all(readable_images)

    assert "测试准确率：98.06%" in document
    assert "测试Macro-F1：0.9790" in document
    assert "不修改原始标签" in document
    assert "独立来源的外部图片测试" in document

    print("\n第五天第六步验证成功")
    print("第五天评估与错误分析成果完整且相互一致")


if __name__ == "__main__":
    main()