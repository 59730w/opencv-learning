import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18_baseline"

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
CLASS_MAP_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "class_to_idx.json"
)

MATRIX_PATH = OUTPUT_DIR / "confusion_matrix_normalized.png"
PAIRS_IMAGE_PATH = OUTPUT_DIR / "confusion_pairs.png"
PAIRS_CSV_PATH = OUTPUT_DIR / "confusion_pairs.csv"


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
        rows = list(csv.DictReader(file))

    y_true = [int(row["true_index"]) for row in rows]
    y_pred = [int(row["predicted_index"]) for row in rows]
    labels = list(range(50))

    counts = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )
    normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )

    # 完整50类归一化混淆矩阵
    blue_cmap = sns.light_palette("#2F6B9A", as_cmap=True)

    fig, ax = plt.subplots(figsize=(20, 17))
    sns.heatmap(
        normalized,
        cmap=blue_cmap,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Recall-normalized proportion"},
        ax=ax,
    )
    ax.set_title(
        "Normalized confusion matrix",
        fontsize=18,
        pad=28,
    )
    ax.text(
        0.5,
        1.01,
        "BarkVN-50 test set · 823 images · rows=true · columns=predicted",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#555555",
    )
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 统计所有非对角线错误类别对
    pair_counter = Counter(
        (true_index, pred_index)
        for true_index, pred_index in zip(y_true, y_pred)
        if true_index != pred_index
    )

    pair_rows = []
    for (true_index, pred_index), count in pair_counter.most_common():
        pair_rows.append(
            {
                "true_index": true_index,
                "true_class": idx_to_class[true_index],
                "predicted_index": pred_index,
                "predicted_class": idx_to_class[pred_index],
                "count": count,
            }
        )

    with PAIRS_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "true_index",
                "true_class",
                "predicted_index",
                "predicted_class",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    # 错误方向横向条形图
    displayed_pairs = pair_rows[:12]
    pair_labels = [
        f"{row['true_class']} → {row['predicted_class']}"
        for row in displayed_pairs
    ]
    pair_counts = [row["count"] for row in displayed_pairs]

    fig, ax = plt.subplots(figsize=(13, 7))
    positions = list(range(len(displayed_pairs)))

    ax.barh(
        positions,
        pair_counts,
        color="#2F6B9A",
        edgecolor="#173B56",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(pair_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(pair_counts) + 1)
    ax.set_xlabel("Misclassified test images")
    ax.set_title(
        "Most frequent class confusions",
        fontsize=16,
        pad=24,
    )
    ax.text(
        0.5,
        1.01,
        "BarkVN-50 test set · 16 errors in 823 images",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#555555",
    )
    ax.grid(axis="x", alpha=0.25)

    for position, count in zip(positions, pair_counts):
        ax.text(
            count + 0.05,
            position,
            str(count),
            va="center",
        )

    fig.subplots_adjust(left=0.38, right=0.96, top=0.86)
    fig.savefig(PAIRS_IMAGE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    correct = int(counts.diagonal().sum())
    errors = int(counts.sum() - correct)

    print("测试图片:", int(counts.sum()))
    print("预测正确:", correct)
    print("预测错误:", errors)
    print("不同错误类别对:", len(pair_rows))
    print("\n最常见的错误方向：")

    for row in pair_rows[:10]:
        print(
            f"{row['true_class']} → "
            f"{row['predicted_class']}: "
            f"{row['count']}张"
        )

    print("\n归一化混淆矩阵:", MATRIX_PATH)
    print("错误类别对图:", PAIRS_IMAGE_PATH)
    print("错误类别对清单:", PAIRS_CSV_PATH)

    assert counts.shape == (50, 50)
    assert counts.sum() == 823
    assert correct == 807
    assert errors == 16
    assert sum(row["count"] for row in pair_rows) == 16
    assert MATRIX_PATH.is_file()
    assert PAIRS_IMAGE_PATH.is_file()
    assert PAIRS_CSV_PATH.is_file()

    print("第五天第三步验证成功")


if __name__ == "__main__":
    main()