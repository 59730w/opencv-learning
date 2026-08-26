from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


RUNTIME_ATOL = 1.0e-4
RUNTIME_RTOL = 1.0e-4
MAX_END_TO_END_LOGIT_MAE = 0.30
MAX_END_TO_END_LOGIT_ERROR = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--day53-code", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    return parser.parse_args()


def read_class_names(class_map_path: Path) -> list[str]:
    with class_map_path.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)
    class_names = [""] * len(class_to_idx)
    for class_name, class_index in class_to_idx.items():
        class_names[class_index] = class_name
    if any(not class_name for class_name in class_names):
        raise ValueError("class indices must be continuous from zero")
    return class_names


def top3_indices(logits: np.ndarray) -> np.ndarray:
    return np.argsort(-logits, axis=1)[:, :3]


def main() -> int:
    args = parse_args()
    if not args.executable.is_file():
        raise FileNotFoundError(f"executable does not exist: {args.executable}")
    if len(args.images) < 3:
        raise ValueError("at least three images are required for batch=1 and batch=3")

    sys.path.insert(0, str(args.day53_code.resolve()))
    from day53_common import build_eval_transform

    class_names = read_class_names(args.class_map)
    transform = build_eval_transform()
    session = ort.InferenceSession(
        str(args.model),
        providers=["CPUExecutionProvider"],
    )

    with tempfile.TemporaryDirectory(prefix="day54_inference_") as temporary_directory:
        output_root = Path(temporary_directory)

        for batch_size in (1, 3):
            selected_images = args.images[:batch_size]
            tensor_path = output_root / f"batch{batch_size}_tensor.bin"
            logits_path = output_root / f"batch{batch_size}_logits.bin"

            command = [
                str(args.executable),
                "--model",
                str(args.model),
                "--class-map",
                str(args.class_map),
                "--tensor-output",
                str(tensor_path),
                "--logits-output",
                str(logits_path),
                "--images",
                *(str(path) for path in selected_images),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"C++ inference failed for batch={batch_size}:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            for expected_text in (
                f"Batch size: {batch_size}",
                f"Input shape: [{batch_size}, 3, 224, 224]",
                f"Output shape: [{batch_size}, 50]",
                "DAY54_CPP_INFERENCE_OK",
            ):
                if expected_text not in completed.stdout:
                    raise AssertionError(
                        f"missing output for batch={batch_size}: {expected_text}"
                    )

            cpp_tensor = np.fromfile(tensor_path, dtype=np.float32).reshape(
                batch_size,
                3,
                224,
                224,
            )
            cpp_logits = np.fromfile(logits_path, dtype=np.float32).reshape(
                batch_size,
                len(class_names),
            )

            python_same_tensor_logits = session.run(
                ["logits"],
                {"images": cpp_tensor},
            )[0]
            runtime_difference = np.abs(cpp_logits - python_same_tensor_logits)
            runtime_max = float(runtime_difference.max())
            runtime_mean = float(runtime_difference.mean())
            if not np.allclose(
                cpp_logits,
                python_same_tensor_logits,
                atol=RUNTIME_ATOL,
                rtol=RUNTIME_RTOL,
            ):
                raise AssertionError(
                    f"same-tensor runtime logits differ for batch={batch_size}: "
                    f"mean={runtime_mean}, max={runtime_max}"
                )

            python_pil_tensor = np.stack(
                [
                    transform(Image.open(path).convert("RGB")).numpy()
                    for path in selected_images
                ]
            )
            python_pil_logits = session.run(
                ["logits"],
                {"images": python_pil_tensor},
            )[0]
            end_to_end_difference = np.abs(cpp_logits - python_pil_logits)
            end_to_end_mean = float(end_to_end_difference.mean())
            end_to_end_max = float(end_to_end_difference.max())
            if end_to_end_mean > MAX_END_TO_END_LOGIT_MAE:
                raise AssertionError(
                    f"end-to-end logit MAE exceeds limit for batch={batch_size}: "
                    f"{end_to_end_mean}"
                )
            if end_to_end_max > MAX_END_TO_END_LOGIT_ERROR:
                raise AssertionError(
                    f"end-to-end maximum logit error exceeds limit for batch={batch_size}: "
                    f"{end_to_end_max}"
                )

            cpp_top3 = top3_indices(cpp_logits)
            same_tensor_top3 = top3_indices(python_same_tensor_logits)
            pil_top3 = top3_indices(python_pil_logits)
            if not np.array_equal(cpp_top3, same_tensor_top3):
                raise AssertionError(
                    f"same-tensor ordered Top-3 differs for batch={batch_size}"
                )
            if not np.array_equal(cpp_top3[:, 0], pil_top3[:, 0]):
                raise AssertionError(
                    f"end-to-end Top-1 differs for batch={batch_size}"
                )
            if any(
                set(cpp_row.tolist()) != set(pil_row.tolist())
                for cpp_row, pil_row in zip(cpp_top3, pil_top3)
            ):
                raise AssertionError(
                    f"end-to-end Top-3 class set differs for batch={batch_size}"
                )

            for image_index, indices in enumerate(cpp_top3):
                expected_top1 = class_names[int(indices[0])]
                marker = f"Image {image_index} Top-1: {expected_top1}"
                if marker not in completed.stdout:
                    raise AssertionError(f"missing class mapping output: {marker}")

            ordered_end_to_end = bool(np.array_equal(cpp_top3, pil_top3))
            print(
                f"batch={batch_size}: runtime_mae={runtime_mean:.8f}, "
                f"runtime_max={runtime_max:.8f}, "
                f"end_to_end_mae={end_to_end_mean:.8f}, "
                f"end_to_end_max={end_to_end_max:.8f}, "
                f"top1_equal=True, top3_set_equal=True, "
                f"ordered_top3_equal={ordered_end_to_end}"
            )

    print("DAY54_INFERENCE_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
