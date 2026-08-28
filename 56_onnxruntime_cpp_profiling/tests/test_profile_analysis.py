import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "code" / "day56_analyze_profile.py"


def load_module():
    spec = importlib.util.spec_from_file_location("day56_analyze_profile", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load analysis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProfileAnalysisTests(unittest.TestCase):
    def test_groups_only_node_operator_events(self):
        module = load_module()
        events = [
            {"cat": "Session", "name": "model_loading_uri", "dur": 100},
            {"cat": "Node", "name": "conv_kernel_time", "dur": 30,
             "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"}},
            {"cat": "Node", "name": "conv_kernel_time", "dur": 70,
             "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"}},
            {"cat": "Node", "name": "relu_kernel_time", "dur": 20,
             "args": {"op_name": "Relu", "provider": "CPUExecutionProvider"}},
            {"cat": "Node", "name": "conv_fence_before", "dur": 999,
             "args": {"op_name": "Conv"}},
        ]
        rows = module.summarize_events(events)
        self.assertEqual([row["op_name"] for row in rows], ["Conv", "Relu"])
        self.assertEqual(rows[0]["event_count"], 2)
        self.assertEqual(rows[0]["total_duration_us"], 100.0)
        self.assertAlmostEqual(rows[0]["share_percent"], 100.0 * 100.0 / 120.0)

    def test_rejects_profile_without_operator_events(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "operator events"):
            module.summarize_events([{"cat": "Session", "dur": 1}])


if __name__ == "__main__":
    unittest.main()
