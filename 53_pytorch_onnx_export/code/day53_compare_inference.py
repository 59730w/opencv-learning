from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from day53_common import (
    build_model_from_checkpoint,
    load_image_batch,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX Runtime logits on real images."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def topk(logits: np.ndarray, k: int = 3) -> np.ndarray:
    return np.argsort(logits, axis=1)[:, ::-1][:, :k]


def main() -> None:
    args = parse_args()
    if not args.onnx_model.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {args.onnx_model}")

    model, class_names, _ = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        class_map_path=args.class_map,
    )
    image_batch = load_image_batch(args.images)

    with torch.inference_mode():
        pytorch_logits = model(image_batch).cpu().numpy()

    session = ort.InferenceSession(
        str(args.onnx_model),
        providers=["CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    onnx_logits = session.run(
        [model_output.name],
        {model_input.name: image_batch.numpy()},
    )[0]

    if pytorch_logits.shape != onnx_logits.shape:
        raise AssertionError(
            f"output shape mismatch: {pytorch_logits.shape} != {onnx_logits.shape}"
        )

    absolute_difference = np.abs(pytorch_logits - onnx_logits)
    max_abs_difference = float(absolute_difference.max())
    mean_abs_difference = float(absolute_difference.mean())
    outputs_close = bool(
        np.allclose(
            pytorch_logits,
            onnx_logits,
            atol=args.atol,
            rtol=args.rtol,
        )
    )

    pytorch_top3 = topk(pytorch_logits, k=3)
    onnx_top3 = topk(onnx_logits, k=3)
    top1_equal = bool(np.array_equal(pytorch_top3[:, 0], onnx_top3[:, 0]))
    top3_equal = bool(np.array_equal(pytorch_top3, onnx_top3))

    predictions = []
    for row, image_path in enumerate(args.images):
        indices = onnx_top3[row].tolist()
        predictions.append(
            {
                "image": image_path.name,
                "image_sha256": sha256_file(image_path),
                "pytorch_top3_indices": pytorch_top3[row].tolist(),
                "onnx_top3_indices": indices,
                "onnx_top3_classes": [class_names[index] for index in indices],
            }
        )

    report = {
        "onnx_model": str(args.onnx_model.resolve()),
        "provider": session.get_providers()[0],
        "batch_shape": list(image_batch.shape),
        "output_shape": list(onnx_logits.shape),
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_difference": max_abs_difference,
        "mean_abs_difference": mean_abs_difference,
        "outputs_close": outputs_close,
        "top1_equal": top1_equal,
        "top3_equal": top3_equal,
        "predictions": predictions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Batch: {tuple(image_batch.shape)}")
    print(f"Output: {onnx_logits.shape}")
    print(f"Provider: {report['provider']}")
    print(f"Max absolute difference: {max_abs_difference:.8f}")
    print(f"Mean absolute difference: {mean_abs_difference:.8f}")
    for item in predictions:
        print(f"{item['image']}: {' | '.join(item['onnx_top3_classes'])}")

    if not outputs_close:
        raise AssertionError("PyTorch and ONNX Runtime outputs exceed tolerance")
    if not top1_equal or not top3_equal:
        raise AssertionError("PyTorch and ONNX Runtime Top-k predictions differ")
    print("DAY53_ONNX_COMPARE_OK")


if __name__ == "__main__":
    main()
