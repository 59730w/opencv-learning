import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "resnet18_baseline"
    / "training_history.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "resnet18_baseline"
    / "training_curves.png"
)


def main():
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train"]["loss"] for record in history]
    val_loss = [record["validation"]["loss"] for record in history]
    train_acc = [record["train"]["accuracy"] for record in history]
    val_acc = [record["validation"]["accuracy"] for record in history]
    val_f1 = [
        record["validation"]["macro_f1"]
        for record in history
    ]

    best_record = max(
        history,
        key=lambda record: record["validation"]["macro_f1"],
    )
    best_epoch = best_record["epoch"]
    best_f1 = best_record["validation"]["macro_f1"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        epochs,
        train_loss,
        marker="o",
        label="Train loss",
    )
    axes[0].plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation loss",
    )
    axes[0].axvline(
        3.5,
        color="gray",
        linestyle="--",
        label="Start fine-tuning",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_acc,
        marker="o",
        label="Train accuracy",
    )
    axes[1].plot(
        epochs,
        val_acc,
        marker="o",
        label="Validation accuracy",
    )
    axes[1].plot(
        epochs,
        val_f1,
        marker="o",
        label="Validation macro-F1",
    )
    axes[1].axvline(
        3.5,
        color="gray",
        linestyle="--",
        label="Start fine-tuning",
    )
    axes[1].scatter(
        [best_epoch],
        [best_f1],
        color="red",
        s=70,
        zorder=5,
        label=f"Best epoch: {best_epoch}",
    )
    axes[1].set_title("Accuracy and macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("ResNet18 baseline training history")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("训练轮数:", len(history))
    print("最佳轮次:", best_epoch)
    print("最佳验证准确率:", f"{best_record['validation']['accuracy']:.2%}")
    print("最佳验证Macro-F1:", f"{best_f1:.4f}")
    print("训练曲线:", OUTPUT_PATH)

    assert len(history) == 15
    assert best_epoch == 13
    assert OUTPUT_PATH.is_file()
    assert OUTPUT_PATH.stat().st_size > 0

    print("第四天第十步验证成功")


if __name__ == "__main__":
    main()