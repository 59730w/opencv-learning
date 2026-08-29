from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_calibration_records(
    rows: list[dict[str, str]], images_per_class: int = 4, expected_classes: int = 50
) -> list[dict[str, str]]:
    if images_per_class <= 0:
        raise ValueError("images_per_class must be positive")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "train":
            grouped[int(row["class_index"])].append(row)
    if sorted(grouped) != list(range(expected_classes)):
        raise ValueError("training split does not contain the expected continuous classes")
    selected: list[dict[str, str]] = []
    for class_index in range(expected_classes):
        candidates = sorted(grouped[class_index], key=lambda row: row["relative_path"])
        if len(candidates) < images_per_class:
            raise ValueError(f"class {class_index} has too few training images")
        selected.extend(candidates[:images_per_class])
    return selected


def preprocess_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected an uint8 BGR image")
    height, width = image.shape[:2]
    if width < height:
        resized_width, resized_height = 256, 256 * height // width
    else:
        resized_width, resized_height = 256 * width // height, 256
    shrinking = resized_width < width or resized_height < height
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )
    left = (resized_width - 224) // 2
    top = (resized_height - 224) // 2
    crop = resized[top : top + 224, left : left + 224]
    tensor = crop.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray((tensor - MEAN) / STD, dtype=np.float32)


class ForestCalibrationReader(CalibrationDataReader):
    def __init__(self, records: list[dict[str, str]], data_root: Path):
        self.records = records
        self.data_root = data_root
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.records):
            return None
        path = self.data_root / Path(self.records[self.index]["relative_path"])
        self.index += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not decode calibration image: {path}")
        return {"images": preprocess_bgr(image)[None, ...]}

    def rewind(self) -> None:
        self.index = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Day57 QDQ S8S8 静态量化模型")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preprocessed-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--images-per-class", type=int, default=4)
    args = parser.parse_args()

    for path in (args.model, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    selected = select_calibration_records(
        read_manifest(args.manifest), images_per_class=args.images_per_class
    )
    if any(row["split"] != "train" for row in selected):
        raise RuntimeError("non-training sample entered calibration")
    for path in (args.preprocessed_model, args.output, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)

    quant_pre_process(args.model, args.preprocessed_model)
    reader = ForestCalibrationReader(selected, args.data_root)
    quantize_static(
        args.preprocessed_model,
        args.output,
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "Gemm"],
        extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
    )

    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    counts = Counter(node.op_type for node in model.graph.node)
    if counts["QuantizeLinear"] == 0 or counts["DequantizeLinear"] == 0:
        raise RuntimeError("QDQ nodes were not generated")
    if counts["Conv"] != 20 or counts["Gemm"] != 1:
        raise RuntimeError("expected Conv/Gemm structure was not preserved")
    onnx.helper.set_model_props(
        model,
        {
            "lesson": "Day57 static INT8 post-training quantization",
            "format": "QDQ S8S8 per-channel weights",
            "calibration": "200 deterministic training-only BarkVN-50 images",
            "limitation": "deployment experiment; does not improve external generalization",
        },
    )
    onnx.save(model, args.output)
    summary = {
        "reference_model_sha256": sha256_file(args.model),
        "reference_model_size_bytes": args.model.stat().st_size,
        "quantized_model_sha256": sha256_file(args.output),
        "quantized_model_size_bytes": args.output.stat().st_size,
        "size_reduction_percent": 100.0 * (1.0 - args.output.stat().st_size / args.model.stat().st_size),
        "calibration_split": "train",
        "calibration_images": len(selected),
        "calibration_classes": len({row["class_index"] for row in selected}),
        "node_counts": dict(sorted(counts.items())),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Calibration images: {len(selected)}")
    print(f"Calibration classes: {summary['calibration_classes']}")
    print(f"Q/DQ nodes: {counts['QuantizeLinear']}/{counts['DequantizeLinear']}")
    print(f"Size reduction: {summary['size_reduction_percent']:.3f}%")
    print("DAY57_CALIBRATION_SPLIT_OK")
    print("DAY57_STATIC_INT8_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
