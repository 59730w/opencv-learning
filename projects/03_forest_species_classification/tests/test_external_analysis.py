import importlib
import importlib.util


def test_find_adjacent_cross_split_pairs_detects_cross_split_neighbors():
    assert importlib.util.find_spec("src.external_analysis") is not None
    module = importlib.import_module("src.external_analysis")
    rows = [
        {
            "relative_path": "Acacia/IMG_10.JPG",
            "class_name": "Acacia",
            "split": "train",
        },
        {
            "relative_path": "Acacia/IMG_11.JPG",
            "class_name": "Acacia",
            "split": "test",
        },
        {
            "relative_path": "Acacia/IMG_20.JPG",
            "class_name": "Acacia",
            "split": "validation",
        },
    ]

    pairs = module.find_adjacent_cross_split_pairs(rows, max_gap=2)

    assert len(pairs) == 1
    assert pairs[0]["class_name"] == "Acacia"
    assert pairs[0]["first_path"] == "Acacia/IMG_10.JPG"
    assert pairs[0]["second_path"] == "Acacia/IMG_11.JPG"
    assert pairs[0]["number_gap"] == 1


def test_render_external_report_preserves_strict_and_posthoc_labels():
    module = importlib.import_module("src.external_analysis")
    assert hasattr(module, "render_external_report")
    evidence = {
        "official_test": {
            "images": 823,
            "accuracy": 0.9805589307411907,
            "macro_f1": 0.9789708352372489,
        },
        "strict": {
            "positive_images": 20,
            "top1_correct": 1,
            "top1_accuracy": 0.05,
            "top3_correct": 4,
            "top3_accuracy": 0.20,
        },
        "posthoc": {
            "strict_external_test": False,
            "top1_correct": 2,
            "top1_accuracy": 0.10,
            "top3_correct": 3,
            "top3_accuracy": 0.15,
        },
        "sequence_risk": {
            "adjacent_cross_split_pairs": 2442,
            "classes_involved": 50,
        },
        "view_metrics": {},
        "negative": {},
        "prediction_counts": {},
        "class_metrics": {},
    }

    text = module.render_external_report(evidence)

    assert "严格外部测试" in text
    assert "| Top-1 | 1/20 | 5.00% |" in text
    assert "post-hoc" in text
    assert "不是新的严格外部测试" in text
    assert "2,442" in text
