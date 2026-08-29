from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day57_quantize import preprocess_bgr, read_manifest


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict[str, float]:
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("labels and predictions must be non-empty and aligned")
    f1_values = []
    for label in range(class_count):
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {"accuracy": float(np.mean(y_true == y_pred)), "macro_f1": float(np.mean(f1_values))}


def run_paths(session: ort.InferenceSession, paths: list[Path], batch_size: int = 16) -> np.ndarray:
    outputs = []
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start : start + batch_size]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"could not decode: {path}")
            tensors.append(preprocess_bgr(image))
        batch = np.stack(tensors).astype(np.float32, copy=False)
        logits = session.run(["logits"], {"images": batch})[0]
        if logits.shape != (len(tensors), 50) or not np.isfinite(logits).all():
            raise RuntimeError("invalid logits")
        outputs.append(logits)
    return np.concatenate(outputs)


def compare_split(
    fp32: ort.InferenceSession,
    int8: ort.InferenceSession,
    records: list[dict[str, str]],
    data_root: Path,
    split: str,
) -> dict:
    chosen = [row for row in records if row["split"] == split]
    paths = [data_root / Path(row["relative_path"]) for row in chosen]
    labels = np.array([int(row["class_index"]) for row in chosen], dtype=np.int64)
    fp32_logits, int8_logits = run_paths(fp32, paths), run_paths(int8, paths)
    fp32_pred, int8_pred = fp32_logits.argmax(1), int8_logits.argmax(1)
    fp32_metrics = classification_metrics(labels, fp32_pred, 50)
    int8_metrics = classification_metrics(labels, int8_pred, 50)
    fp32_top3 = np.argsort(-fp32_logits, axis=1)[:, :3]
    int8_top3 = np.argsort(-int8_logits, axis=1)[:, :3]
    return {
        "images": len(chosen),
        "fp32": fp32_metrics,
        "int8": int8_metrics,
        "accuracy_drop": fp32_metrics["accuracy"] - int8_metrics["accuracy"],
        "macro_f1_drop": fp32_metrics["macro_f1"] - int8_metrics["macro_f1"],
        "top1_agreement": float(np.mean(fp32_pred == int8_pred)),
        "ordered_top3_agreement": float(np.mean(np.all(fp32_top3 == int8_top3, axis=1))),
        "mean_absolute_logit_difference": float(np.mean(np.abs(fp32_logits - int8_logits))),
        "maximum_absolute_logit_difference": float(np.max(np.abs(fp32_logits - int8_logits))),
    }


def softmax_confidence(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.max(axis=1)


def compare_external(
    fp32: ort.InferenceSession,
    int8: ort.InferenceSession,
    metadata_path: Path,
    external_root: Path,
    class_map_path: Path,
) -> dict:
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    paths = [external_root / row["test_type"] / row["filename"] for row in rows]
    fp32_logits, int8_logits = run_paths(fp32, paths), run_paths(int8, paths)
    fp32_pred, int8_pred = fp32_logits.argmax(1), int8_logits.argmax(1)
    positive_indices = [i for i, row in enumerate(rows) if row["test_type"] == "positive"]
    negative_indices = [i for i, row in enumerate(rows) if row["test_type"] == "negative"]
    expected = np.array([class_map[rows[i]["expected_class"]] for i in positive_indices])
    return {
        "images": len(rows),
        "positive_images": len(positive_indices),
        "negative_images": len(negative_indices),
        "fp32_positive_accuracy": float(np.mean(fp32_pred[positive_indices] == expected)),
        "int8_positive_accuracy": float(np.mean(int8_pred[positive_indices] == expected)),
        "all_top1_agreement": float(np.mean(fp32_pred == int8_pred)),
        "negative_top1_agreement": float(np.mean(fp32_pred[negative_indices] == int8_pred[negative_indices])),
        "fp32_negative_mean_max_confidence": float(np.mean(softmax_confidence(fp32_logits[negative_indices]))),
        "int8_negative_mean_max_confidence": float(np.mean(softmax_confidence(int8_logits[negative_indices]))),
        "warning": "Frozen external comparison only; not used for calibration or quantization selection.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 Day57 FP32/INT8 内部质量与冻结外部表现")
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--external-metadata", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    fp32 = ort.InferenceSession(str(args.fp32), sess_options=options, providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(args.int8), sess_options=options, providers=["CPUExecutionProvider"])
    records = read_manifest(args.manifest)
    report = {"validation": compare_split(fp32, int8, records, args.data_root, "validation")}
    validation = report["validation"]
    gate_pass = validation["accuracy_drop"] <= 0.01 and validation["macro_f1_drop"] <= 0.01
    report["validation_gate_pass"] = gate_pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not gate_pass:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("DAY57_INTERNAL_QUALITY_GATE_FAILED")
        return 6
    report["test"] = compare_split(fp32, int8, records, args.data_root, "test")
    report["external"] = compare_external(
        fp32, int8, args.external_metadata, args.external_root, args.class_map
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("DAY57_INTERNAL_QUALITY_GATE_OK")
    print("DAY57_EXTERNAL_COMPARISON_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
