import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ManifestImageDataset
from src.inference import average_crop_probabilities
from src.model import build_resnet18
from src.transforms import build_five_crop_eval_transform


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
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_baseline" / "best.pt"
EXTERNAL_METADATA_PATH = PROJECT_ROOT / "external_images" / "metadata.csv"
BASELINE_EXTERNAL_METRICS_PATH = PROJECT_ROOT / "outputs" / "external_test" / "external_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_test_posthoc_multicrop"
METRICS_PATH = OUTPUT_DIR / "multicrop_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "multicrop_external_predictions.csv"


class ExternalFiveCropDataset(Dataset):
    def __init__(self, metadata_rows, transform):
        self.metadata_rows = metadata_rows
        self.transform = transform

    def __len__(self):
        return len(self.metadata_rows)

    def __getitem__(self, index):
        row = self.metadata_rows[index]
        path = PROJECT_ROOT / "external_images" / row["test_type"] / row["filename"]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            crops = self.transform(image)
        return crops, index


def load_external_metadata():
    with EXTERNAL_METADATA_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 26:
        raise ValueError(f"外部测试清单应为26张，实际为{len(rows)}张")
    return rows


def predict_five_crop(model, crops, device, use_amp):
    batch_size, crop_count, channels, height, width = crops.shape
    crops = crops.reshape(
        batch_size * crop_count,
        channels,
        height,
        width,
    ).to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        logits = model(crops)
    logits = logits.reshape(batch_size, crop_count, -1)
    return average_crop_probabilities(logits)


def evaluate_validation(model, device, use_amp, class_count):
    dataset = ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split="validation",
        transform=build_five_crop_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    labels_all = []
    predictions_all = []
    with torch.inference_mode():
        for crops, labels, _ in loader:
            probabilities = predict_five_crop(model, crops, device, use_amp)
            predictions = probabilities.argmax(dim=1)
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.cpu().tolist())
    return {
        "images": len(dataset),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(
            labels_all,
            predictions_all,
            labels=list(range(class_count)),
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_external(model, device, use_amp, metadata_rows, idx_to_class):
    dataset = ExternalFiveCropDataset(
        metadata_rows,
        build_five_crop_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows_by_index = {}
    with torch.inference_mode():
        for crops, indices in loader:
            probabilities = predict_five_crop(model, crops, device, use_amp)
            top_probabilities, top_indices = probabilities.topk(3, dim=1)
            for item_index, probs, class_indices in zip(
                indices.tolist(),
                top_probabilities.cpu().tolist(),
                top_indices.cpu().tolist(),
            ):
                metadata = metadata_rows[item_index]
                top_classes = [idx_to_class[index] for index in class_indices]
                is_positive = metadata["test_type"] == "positive"
                rows_by_index[item_index] = {
                    "filename": metadata["filename"],
                    "test_type": metadata["test_type"],
                    "expected_class": metadata["expected_class"],
                    "view_type": metadata["view_type"],
                    "top1_class": top_classes[0],
                    "top1_confidence": probs[0],
                    "top1_correct": (
                        top_classes[0] == metadata["expected_class"] if is_positive else ""
                    ),
                    "top2_class": top_classes[1],
                    "top2_confidence": probs[1],
                    "top3_class": top_classes[2],
                    "top3_confidence": probs[2],
                    "expected_in_top3": (
                        metadata["expected_class"] in top_classes if is_positive else ""
                    ),
                }

    prediction_rows = [rows_by_index[index] for index in range(len(metadata_rows))]
    positive_rows = [row for row in prediction_rows if row["test_type"] == "positive"]
    negative_rows = [row for row in prediction_rows if row["test_type"] == "negative"]
    by_view = defaultdict(list)
    for row in positive_rows:
        by_view[row["view_type"]].append(row)

    top1_correct = sum(bool(row["top1_correct"]) for row in positive_rows)
    top3_correct = sum(bool(row["expected_in_top3"]) for row in positive_rows)
    summary = {
        "images": len(prediction_rows),
        "positive": {
            "images": len(positive_rows),
            "top1_correct": top1_correct,
            "top1_accuracy": top1_correct / len(positive_rows),
            "top3_correct": top3_correct,
            "top3_accuracy": top3_correct / len(positive_rows),
            "by_view": {
                view: {
                    "images": len(rows),
                    "top1_correct": sum(bool(row["top1_correct"]) for row in rows),
                    "top3_correct": sum(bool(row["expected_in_top3"]) for row in rows),
                }
                for view, rows in sorted(by_view.items())
            },
        },
        "negative": {
            "images": len(negative_rows),
            "confidence_ge_0_8": sum(row["top1_confidence"] >= 0.8 for row in negative_rows),
            "max_top1_confidence": max(row["top1_confidence"] for row in negative_rows),
        },
    }
    return prediction_rows, summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    class_to_idx = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    metadata_rows = load_external_metadata()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = build_resnet18(
        num_classes=len(class_to_idx),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    validation_metrics = evaluate_validation(
        model,
        device,
        use_amp,
        len(class_to_idx),
    )
    prediction_rows, external_metrics = evaluate_external(
        model,
        device,
        use_amp,
        metadata_rows,
        idx_to_class,
    )
    baseline_external = json.loads(
        BASELINE_EXTERNAL_METRICS_PATH.read_text(encoding="utf-8")
    )

    output = {
        "experiment_type": "post-hoc exploratory inference optimization",
        "strict_external_test": False,
        "reason": "Five-crop averaging was designed after observing the fixed external baseline.",
        "checkpoint_epoch": checkpoint["epoch"],
        "method": "Resize shortest side to 256, take FiveCrop(224), average crop softmax probabilities",
        "validation": validation_metrics,
        "external_baseline_center_crop": baseline_external["positive"],
        "external_multicrop": external_metrics,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with PREDICTIONS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    positive = external_metrics["positive"]
    baseline = baseline_external["positive"]
    print("设备:", device)
    print("检查点轮次:", checkpoint["epoch"])
    print("五裁剪验证集Accuracy:", f"{validation_metrics['accuracy']:.2%}")
    print("五裁剪验证集Macro-F1:", f"{validation_metrics['macro_f1']:.4f}")
    print(
        "外部Top-1（中心裁剪 -> 五裁剪）:",
        f"{baseline['top1_correct']}/20 -> {positive['top1_correct']}/20",
    )
    print(
        "外部Top-3（中心裁剪 -> 五裁剪）:",
        f"{baseline['top3_correct']}/20 -> {positive['top3_correct']}/20",
    )
    for view, metrics in positive["by_view"].items():
        print(
            f"{view}: Top-1 {metrics['top1_correct']}/{metrics['images']}, "
            f"Top-3 {metrics['top3_correct']}/{metrics['images']}"
        )
    print("结果:", METRICS_PATH)
    print("逐图预测:", PREDICTIONS_PATH)
    print("注意：这是看到外部基线后的探索结果，不是新的严格外部测试")

    assert checkpoint["epoch"] == 13
    assert validation_metrics["images"] == 844
    assert external_metrics["images"] == 26
    assert external_metrics["positive"]["images"] == 20
    assert external_metrics["negative"]["images"] == 6
    assert METRICS_PATH.is_file()
    assert PREDICTIONS_PATH.is_file()


if __name__ == "__main__":
    main()
