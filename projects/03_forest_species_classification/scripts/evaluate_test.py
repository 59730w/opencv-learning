import csv
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ManifestImageDataset
from src.model import build_resnet18
from src.transforms import build_eval_transform


DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "BarkVN-50"
    / "v1"
    / "images"
    / "BarkVN-50_mendeley"
)
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "split_manifest.csv"
CLASS_MAP_PATH = PROJECT_ROOT / "datasets" / "processed" / "class_to_idx.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_baseline"
    / "best.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18_baseline"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    class_to_idx = json.loads(
        CLASS_MAP_PATH.read_text(encoding="utf-8")
    )
    idx_to_class = {
        index: name for name, index in class_to_idx.items()
    }

    dataset = ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split="test",
        transform=build_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = build_resnet18(
        num_classes=50,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []
    prediction_rows = []

    with torch.inference_mode():
        for images, labels, relative_paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            labels_cpu = labels.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            confidences_cpu = confidences.cpu().tolist()

            all_labels.extend(labels_cpu)
            all_predictions.extend(predictions_cpu)

            for path, true_idx, pred_idx, confidence in zip(
                relative_paths,
                labels_cpu,
                predictions_cpu,
                confidences_cpu,
            ):
                prediction_rows.append(
                    {
                        "relative_path": path,
                        "true_index": true_idx,
                        "true_class": idx_to_class[true_idx],
                        "predicted_index": pred_idx,
                        "predicted_class": idx_to_class[pred_idx],
                        "confidence": confidence,
                        "correct": true_idx == pred_idx,
                    }
                )

    test_loss = total_loss / total_samples
    test_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )
    test_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(50)),
        average="macro",
        zero_division=0,
    )
    correct_count = sum(
        row["correct"] for row in prediction_rows
    )

    metrics = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_stage": checkpoint["stage"],
        "test_images": total_samples,
        "correct_predictions": correct_count,
        "incorrect_predictions": total_samples - correct_count,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with PREDICTIONS_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "true_index",
                "true_class",
                "predicted_index",
                "predicted_class",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print("设备:", device)
    print("检查点轮次:", checkpoint["epoch"])
    print("测试图片:", total_samples)
    print("预测正确:", correct_count)
    print("预测错误:", total_samples - correct_count)
    print("测试损失:", round(test_loss, 4))
    print("测试准确率:", f"{test_accuracy:.2%}")
    print("测试Macro-F1:", f"{test_macro_f1:.4f}")
    print("测试指标:", METRICS_PATH)
    print("逐图预测:", PREDICTIONS_PATH)

    assert checkpoint["epoch"] == 13
    assert total_samples == 823
    assert len(prediction_rows) == 823
    assert 0 <= test_accuracy <= 1
    assert 0 <= test_macro_f1 <= 1
    assert METRICS_PATH.is_file()
    assert PREDICTIONS_PATH.is_file()

    print("第五天第一步验证成功")


if __name__ == "__main__":
    main()