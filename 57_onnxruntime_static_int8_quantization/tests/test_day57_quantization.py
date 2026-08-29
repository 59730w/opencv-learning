import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantizationTests(unittest.TestCase):
    def test_calibration_selection_is_balanced_deterministic_and_train_only(self):
        module = load_module("day57_quantize", ROOT / "code" / "day57_quantize.py")
        rows = []
        for class_index, class_name in enumerate(("A", "B")):
            for split, count in (("train", 3), ("validation", 2), ("test", 1)):
                for index in reversed(range(count)):
                    rows.append({"relative_path": f"{class_name}/{index}.jpg",
                                 "class_name": class_name,
                                 "class_index": str(class_index), "split": split})
        selected = module.select_calibration_records(rows, images_per_class=2,
                                                     expected_classes=2)
        self.assertEqual([row["relative_path"] for row in selected],
                         ["A/0.jpg", "A/1.jpg", "B/0.jpg", "B/1.jpg"])
        self.assertTrue(all(row["split"] == "train" for row in selected))

    def test_opencv_preprocessing_matches_rgb_normalization_contract(self):
        module = load_module("day57_quantize", ROOT / "code" / "day57_quantize.py")
        bgr = np.zeros((300, 400, 3), dtype=np.uint8)
        bgr[:] = (10, 20, 30)
        tensor = module.preprocess_bgr(bgr)
        expected = (
            np.array([30, 20, 10], dtype=np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[:, 100, 100], expected, atol=1e-6)

    def test_metrics_report_accuracy_and_macro_f1(self):
        module = load_module("day57_evaluate", ROOT / "code" / "day57_evaluate.py")
        metrics = module.classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), class_count=2)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 2)

    def test_benchmark_speedup_uses_fp32_over_int8_latency(self):
        module = load_module("day57_benchmark", ROOT / "code" / "day57_benchmark.py")
        self.assertAlmostEqual(module.speedup(12.0, 8.0), 1.5)
        with self.assertRaises(ValueError):
            module.speedup(12.0, 0.0)


if __name__ == "__main__":
    unittest.main()
