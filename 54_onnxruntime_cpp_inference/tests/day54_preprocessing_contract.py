from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


TENSOR_SHAPE = (3, 224, 224)
MAX_MEAN_ABSOLUTE_ERROR = 0.03
MAX_ABSOLUTE_ERROR = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--day53-code", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.executable.is_file():
        raise FileNotFoundError(f"executable does not exist: {args.executable}")

    sys.path.insert(0, str(args.day53_code.resolve()))
    from day53_common import build_eval_transform

    transform = build_eval_transform()
    with tempfile.TemporaryDirectory(prefix="day54_preprocess_") as temporary_directory:
        output_root = Path(temporary_directory)
        for index, image_path in enumerate(args.images):
            output_path = output_root / f"tensor_{index}.bin"
            completed = subprocess.run(
                [str(args.executable), str(image_path), "--output", str(output_path)],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"C++ preprocessing failed for {image_path}:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            if "DAY54_PREPROCESS_OK" not in completed.stdout:
                raise AssertionError(f"missing success marker for {image_path}")

            actual = np.fromfile(output_path, dtype=np.float32)
            if actual.size != int(np.prod(TENSOR_SHAPE)):
                raise AssertionError(
                    f"unexpected tensor size for {image_path}: {actual.size}"
                )
            actual = actual.reshape(TENSOR_SHAPE)

            with Image.open(image_path) as image:
                reference = transform(image.convert("RGB")).numpy()

            difference = np.abs(actual - reference)
            mean_absolute_error = float(difference.mean())
            max_absolute_error = float(difference.max())
            print(
                f"{image_path.name}: "
                f"mae={mean_absolute_error:.8f}, max={max_absolute_error:.8f}"
            )
            if mean_absolute_error > MAX_MEAN_ABSOLUTE_ERROR:
                raise AssertionError(
                    f"mean absolute error exceeds {MAX_MEAN_ABSOLUTE_ERROR}: "
                    f"{mean_absolute_error}"
                )
            if max_absolute_error > MAX_ABSOLUTE_ERROR:
                raise AssertionError(
                    f"maximum absolute error exceeds {MAX_ABSOLUTE_ERROR}: "
                    f"{max_absolute_error}"
                )

    print("DAY54_PREPROCESS_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
