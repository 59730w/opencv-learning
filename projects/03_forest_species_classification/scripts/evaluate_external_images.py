import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_resnet18
from src.transforms import build_eval_transform


METADATA_PATH = PROJECT_ROOT / "external_images" / "metadata.csv"
CLASS_MAP_PATH = PROJECT_ROOT / "datasets" / "processed" / "class_to_idx.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_baseline" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_test"
PREDICTIONS_PATH = OUTPUT_DIR / "external_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "external_metrics.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "external_prediction_contact_sheet.jpg"
EXPECTED_TOTAL = 26


def read_metadata():
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"外部测试清单应有{EXPECTED_TOTAL}条，实际为{len(rows)}条")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("外部测试清单存在重复文件名")
    return rows


def image_path(row):
    return PROJECT_ROOT / "external_images" / row["test_type"] / row["filename"]


def load_batch(metadata_rows, transform):
    tensors = []
    for row in metadata_rows:
        path = image_path(row)
        if not path.is_file():
            raise FileNotFoundError(f"外部图片不存在：{path}")
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensors.append(transform(image))
    return torch.stack(tensors)


def make_prediction_rows(metadata_rows, probabilities, idx_to_class):
    top_probabilities, top_indices = probabilities.topk(k=3, dim=1)
    prediction_rows = []

    for row, probs, indices in zip(
        metadata_rows,
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        top_classes = [idx_to_class[index] for index in indices]
        is_positive = row["test_type"] == "positive"
        expected_class = row["expected_class"]
        top1_correct = is_positive and top_classes[0] == expected_class
        expected_in_top3 = is_positive and expected_class in top_classes

        prediction_rows.append({
            "filename": row["filename"],
            "test_type": row["test_type"],
            "expected_class": expected_class,
            "view_type": row["view_type"],
            "top1_index": indices[0],
            "top1_class": top_classes[0],
            "top1_confidence": probs[0],
            "top1_correct": top1_correct if is_positive else "",
            "top2_index": indices[1],
            "top2_class": top_classes[1],
            "top2_confidence": probs[1],
            "top3_index": indices[2],
            "top3_class": top_classes[2],
            "top3_confidence": probs[2],
            "expected_in_top3": expected_in_top3 if is_positive else "",
        })
    return prediction_rows


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def summarize_positive(rows):
    top1_correct = sum(bool(row["top1_correct"]) for row in rows)
    top3_correct = sum(bool(row["expected_in_top3"]) for row in rows)

    grouped_by_view = defaultdict(list)
    grouped_by_class = defaultdict(list)
    for row in rows:
        grouped_by_view[row["view_type"]].append(row)
        grouped_by_class[row["expected_class"]].append(row)

    def group_metrics(group):
        count = len(group)
        group_top1 = sum(bool(row["top1_correct"]) for row in group)
        group_top3 = sum(bool(row["expected_in_top3"]) for row in group)
        return {
            "images": count,
            "top1_correct": group_top1,
            "top1_accuracy": ratio(group_top1, count),
            "top3_correct": group_top3,
            "top3_accuracy": ratio(group_top3, count),
        }

    return {
        "images": len(rows),
        "top1_correct": top1_correct,
        "top1_accuracy": ratio(top1_correct, len(rows)),
        "top3_correct": top3_correct,
        "top3_accuracy": ratio(top3_correct, len(rows)),
        "by_view": {
            name: group_metrics(group)
            for name, group in sorted(grouped_by_view.items())
        },
        "by_class": {
            name: group_metrics(group)
            for name, group in sorted(grouped_by_class.items())
        },
    }


def summarize_negative(rows):
    high_confidence = [row for row in rows if row["top1_confidence"] >= 0.8]
    return {
        "images": len(rows),
        "confidence_ge_0_8": len(high_confidence),
        "mean_top1_confidence": sum(row["top1_confidence"] for row in rows) / len(rows),
        "max_top1_confidence": max(row["top1_confidence"] for row in rows),
        "forced_predictions": [
            {
                "filename": row["filename"],
                "negative_type": row["view_type"],
                "predicted_class": row["top1_class"],
                "confidence": row["top1_confidence"],
            }
            for row in rows
        ],
    }


def write_predictions(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(metadata_rows, prediction_rows):
    columns = 4
    cell_width, image_height, label_height = 420, 300, 92
    rows = math.ceil(len(metadata_rows) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (metadata, prediction) in enumerate(zip(metadata_rows, prediction_rows)):
        path = image_path(metadata)
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            preview = ImageOps.contain(source, (cell_width - 16, image_height - 16))

        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + label_height)
        image_x = x + (cell_width - preview.width) // 2
        image_y = y + (image_height - preview.height) // 2
        canvas.paste(preview, (image_x, image_y))

        if metadata["test_type"] == "negative":
            color = "darkorange"
            line2 = f"negative/{metadata['view_type']} -> {prediction['top1_class']}"
        elif prediction["top1_correct"]:
            color = "green"
            line2 = f"correct: {prediction['top1_class']}"
        else:
            color = "red"
            line2 = f"{metadata['expected_class']} -> {prediction['top1_class']}"

        draw.rectangle(
            (x, y, x + cell_width - 1, y + image_height + label_height - 1),
            outline=color,
            width=4,
        )
        draw.text((x + 8, y + image_height + 6), metadata["filename"], fill="black", font=font)
        draw.text((x + 8, y + image_height + 28), line2, fill=color, font=font)
        draw.text(
            (x + 8, y + image_height + 50),
            f"Top-1 {prediction['top1_confidence']:.2%} | Top-3: "
            f"{prediction['top1_class']} / {prediction['top2_class']} / {prediction['top3_class']}",
            fill="black",
            font=font,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(CONTACT_SHEET_PATH, quality=92)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    class_to_idx = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    metadata_rows = read_metadata()

    positive_classes = {row["expected_class"] for row in metadata_rows if row["test_type"] == "positive"}
    unknown_classes = positive_classes - set(class_to_idx)
    if unknown_classes:
        raise ValueError(f"外部测试类别不在训练类别中：{sorted(unknown_classes)}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = build_resnet18(
        num_classes=len(class_to_idx),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    images = load_batch(metadata_rows, build_eval_transform()).to(device)
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1)

    prediction_rows = make_prediction_rows(metadata_rows, probabilities, idx_to_class)
    positive_rows = [row for row in prediction_rows if row["test_type"] == "positive"]
    negative_rows = [row for row in prediction_rows if row["test_type"] == "negative"]
    positive_metrics = summarize_positive(positive_rows)
    negative_metrics = summarize_negative(negative_rows)

    metrics = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_stage": checkpoint["stage"],
        "preprocessing": "Resize(256), CenterCrop(224), ImageNet normalization",
        "external_images": len(metadata_rows),
        "positive": positive_metrics,
        "negative": negative_metrics,
        "limitations": [
            "The classifier is closed-set and must assign every negative image to one of 50 classes.",
            "The fixed external set was not used for training, checkpoint selection, or parameter tuning.",
            "Twenty positive images are a small diagnostic sample, not a population-level benchmark.",
        ],
    }

    write_predictions(prediction_rows)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(metadata_rows, prediction_rows)

    print("设备:", device)
    print("检查点轮次:", checkpoint["epoch"])
    print("外部图片:", len(metadata_rows))
    print("正样本Top-1:", f"{positive_metrics['top1_correct']}/20 ({positive_metrics['top1_accuracy']:.2%})")
    print("正样本Top-3:", f"{positive_metrics['top3_correct']}/20 ({positive_metrics['top3_accuracy']:.2%})")
    for view, view_metrics in positive_metrics["by_view"].items():
        print(
            f"{view}: Top-1 {view_metrics['top1_correct']}/{view_metrics['images']} "
            f"({view_metrics['top1_accuracy']:.2%}), Top-3 "
            f"{view_metrics['top3_correct']}/{view_metrics['images']} "
            f"({view_metrics['top3_accuracy']:.2%})"
        )
    print("负样本:", negative_metrics["images"])
    print("负样本置信度>=80%:", negative_metrics["confidence_ge_0_8"])
    for row in negative_metrics["forced_predictions"]:
        print(
            f"{row['filename']} | {row['negative_type']} -> "
            f"{row['predicted_class']} | {row['confidence']:.2%}"
        )
    print("逐图预测:", PREDICTIONS_PATH)
    print("汇总指标:", METRICS_PATH)
    print("预测联系表:", CONTACT_SHEET_PATH)

    assert checkpoint["epoch"] == 13
    assert checkpoint["stage"] == "finetune"
    assert logits.shape == (EXPECTED_TOTAL, len(class_to_idx))
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(EXPECTED_TOTAL, device=device),
        atol=1e-5,
    )
    assert len(positive_rows) == 20
    assert len(negative_rows) == 6
    assert PREDICTIONS_PATH.is_file()
    assert METRICS_PATH.is_file()
    assert CONTACT_SHEET_PATH.is_file()
    print("第六天第四步外部推理成功")


if __name__ == "__main__":
    main()
