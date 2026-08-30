import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_module():
    path = ROOT / "code" / "day58_thread_experiment.py"
    spec = importlib.util.spec_from_file_location("day58_thread_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Day58ThreadExperimentTests(unittest.TestCase):
    def test_matrix_has_each_model_batch_and_thread_cell_once(self):
        module = load_module()
        matrix = module.experiment_matrix()
        keys = {(row["model"], row["batch_size"], row["intra_threads"]) for row in matrix}
        expected = {
            (model, batch, threads)
            for model in ("fp32", "int8")
            for batch in (1, 6)
            for threads in (0, 1)
        }
        self.assertEqual(keys, expected)
        self.assertEqual(len(matrix), 8)

    def test_matrix_pairs_models_and_balances_which_model_runs_first(self):
        module = load_module()
        matrix = module.experiment_matrix()
        pairs = [matrix[index:index + 2] for index in range(0, len(matrix), 2)]
        for pair in pairs:
            self.assertEqual({row["model"] for row in pair}, {"fp32", "int8"})
            self.assertEqual(len({(row["batch_size"], row["intra_threads"]) for row in pair}), 1)
        self.assertEqual([pair[0]["model"] for pair in pairs], ["fp32", "int8", "int8", "fp32"])

    def test_ratio_requires_positive_values(self):
        module = load_module()
        self.assertAlmostEqual(module.ratio(12.0, 8.0), 1.5)
        for numerator, denominator in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
            with self.assertRaises(ValueError):
                module.ratio(numerator, denominator)

    def test_build_command_fixes_runtime_boundary_and_measurement_counts(self):
        module = load_module()
        command = module.build_command(
            executable=Path("bench.exe"), model=Path("model.onnx"),
            class_map=Path("classes.json"), images=[Path(f"{i}.png") for i in range(6)],
            batch_size=6, intra_threads=1, raw_output=Path("raw.csv"),
            summary_output=Path("summary.csv"), logits_output=Path("logits.bin"),
        )
        joined = " ".join(command)
        self.assertIn("--mode runtime-only", joined)
        self.assertIn("--warmup 5", joined)
        self.assertIn("--runs 30", joined)
        self.assertIn("--intra-threads 1", joined)

    def test_same_model_logits_require_finite_equal_ordered_top3(self):
        module = load_module()
        default = np.arange(50, dtype=np.float32)[None, :]
        single = default.copy()
        single[0, 0] += 0.25
        with tempfile.TemporaryDirectory() as directory:
            default_path = Path(directory) / "default.bin"
            single_path = Path(directory) / "single.bin"
            default.tofile(default_path)
            single.tofile(single_path)
            evidence = module.validate_same_model_logits(default_path, single_path, batch_size=1)
            self.assertEqual(evidence["ordered_top3_equal"], True)
            self.assertAlmostEqual(evidence["maximum_absolute_difference"], 0.25)
            single[0, 49] = np.nan
            single.tofile(single_path)
            with self.assertRaises(ValueError):
                module.validate_same_model_logits(default_path, single_path, batch_size=1)

    def test_enrich_rows_computes_same_thread_speedup_and_thread_ratio(self):
        module = load_module()
        medians = {
            ("fp32", 1, 0): 10.0, ("int8", 1, 0): 12.0,
            ("fp32", 1, 1): 20.0, ("int8", 1, 1): 15.0,
            ("fp32", 6, 0): 60.0, ("int8", 6, 0): 75.0,
            ("fp32", 6, 1): 90.0, ("int8", 6, 1): 80.0,
        }
        rows = [
            {"model": model, "batch_size": str(batch), "intra_threads": str(threads),
             "median_ms": str(median), "p90_ms": str(median + 1),
             "images_per_second": "1.0"}
            for (model, batch, threads), median in medians.items()
        ]
        enriched, evidence = module.enrich_rows(rows)
        by_key = {
            (row["model"], int(row["batch_size"]), int(row["intra_threads"])): row
            for row in enriched
        }
        self.assertAlmostEqual(float(by_key[("int8", 1, 0)]["speedup_vs_fp32_same_threads"]), 10 / 12)
        self.assertAlmostEqual(float(by_key[("int8", 1, 1)]["speedup_vs_fp32_same_threads"]), 20 / 15)
        self.assertAlmostEqual(float(by_key[("fp32", 1, 1)]["default_vs_single_thread_ratio"]), 10 / 20)
        self.assertEqual(evidence["interpretation"], "full_support")


if __name__ == "__main__":
    unittest.main()
