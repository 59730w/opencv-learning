from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from day62_morphology_regions import (
    acceptance_checks,
    apply_morphology,
    choose_candidate,
    clean_candidate_mask,
    component_records,
    contour_records,
    evaluate_configurations,
    filter_components_by_area,
    filter_components_perspective,
    region_change_metrics,
    select_v2_from_fold_summaries,
    split_cv_folds,
    split_development_stems,
    validate_binary_mask,
    v2_acceptance_checks,
    vertical_line_support_metrics,
)


def test_validate_binary_mask_rejects_wrong_dtype_and_nonbinary_values() -> None:
    with pytest.raises(ValueError, match="uint8"):
        validate_binary_mask(np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="0 or 255"):
        validate_binary_mask(np.full((8, 8), 17, dtype=np.uint8))


def test_open_then_close_removes_speck_and_fills_hole_without_mutating_input() -> None:
    mask = np.zeros((31, 31), dtype=np.uint8)
    cv2.rectangle(mask, (8, 8), (22, 22), 255, -1)
    mask[15, 15] = 0
    mask[3, 3] = 255
    original = mask.copy()

    cleaned = apply_morphology(
        mask,
        open_kernel=3,
        close_kernel=3,
        order="open_close",
    )

    assert np.array_equal(mask, original)
    assert cleaned.dtype == np.uint8
    assert set(np.unique(cleaned)).issubset({0, 255})
    assert cleaned[3, 3] == 0
    assert cleaned[15, 15] == 255


def test_component_records_excludes_background_and_sorts_largest_first() -> None:
    mask = np.zeros((30, 40), dtype=np.uint8)
    cv2.rectangle(mask, (2, 3), (5, 7), 255, -1)
    cv2.rectangle(mask, (20, 10), (29, 19), 255, -1)

    records = component_records(mask)

    assert [record["area"] for record in records] == [100, 20]
    assert records[0]["bbox"] == [20, 10, 10, 10]
    assert records[1]["centroid"] == pytest.approx([3.5, 5.0])


def test_contour_records_report_area_perimeter_and_bounding_box() -> None:
    mask = np.zeros((30, 40), dtype=np.uint8)
    cv2.rectangle(mask, (5, 6), (14, 15), 255, -1)

    records = contour_records(mask)

    assert len(records) == 1
    assert records[0]["area"] == pytest.approx(81.0)
    assert records[0]["perimeter"] == pytest.approx(36.0)
    assert records[0]["bbox"] == [5, 6, 10, 10]


def test_area_filter_removes_small_components_and_keeps_large_component() -> None:
    mask = np.zeros((30, 40), dtype=np.uint8)
    cv2.rectangle(mask, (2, 3), (4, 5), 255, -1)
    cv2.rectangle(mask, (20, 10), (29, 19), 255, -1)

    filtered = filter_components_by_area(mask, min_area=20)

    assert filtered[4, 3] == 0
    assert filtered[15, 25] == 255
    assert set(np.unique(filtered)).issubset({0, 255})


def test_region_change_metrics_report_fragment_reduction_and_pixel_changes() -> None:
    raw = np.zeros((20, 20), dtype=np.uint8)
    cv2.rectangle(raw, (5, 5), (14, 14), 255, -1)
    raw[1, 1] = 255
    cleaned = raw.copy()
    cleaned[1, 1] = 0
    cleaned[9, 9] = 0
    cleaned[9, 9] = 255

    metrics = region_change_metrics(raw, cleaned, small_area=10)

    assert metrics["raw_component_count"] == 2
    assert metrics["cleaned_component_count"] == 1
    assert metrics["component_reduction_fraction"] == pytest.approx(0.5)
    assert metrics["removed_raw_foreground_fraction"] == pytest.approx(1 / 101)
    assert metrics["added_image_fraction"] == 0.0


def test_development_split_is_deterministic_disjoint_and_complete() -> None:
    stems = [str(index) for index in range(1250)]

    first = split_development_stems(stems)
    second = split_development_stems(stems)

    assert first == second
    assert {name: len(items) for name, items in first.items()} == {
        "search": 256,
        "review": 256,
        "stress": 738,
    }
    assert set(first["search"]).isdisjoint(first["review"])
    assert set(first["search"]).isdisjoint(first["stress"])
    assert set(first["review"]).isdisjoint(first["stress"])
    assert set().union(*map(set, first.values())) == set(stems)


def test_clean_candidate_mask_converts_area_fraction_to_pixel_threshold() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (2, 2), (5, 5), 255, -1)  # 16 pixels
    cv2.rectangle(mask, (30, 30), (49, 49), 255, -1)  # 400 pixels
    config = {
        "name": "area_filter",
        "order": None,
        "open_kernel": 3,
        "close_kernel": 5,
        "min_area_fraction": 0.002,
    }

    cleaned = clean_candidate_mask(mask, config)

    assert cleaned[3, 3] == 0
    assert cleaned[40, 40] == 255


def test_acceptance_checks_require_proxy_preservation_and_fragment_reduction() -> None:
    baseline = {
        "mean_support": 0.44,
        "mean_gap": 0.25,
        "p10_gap": 0.15,
        "max_off_line_activation": 0.39,
    }
    accepted = {
        "mean_support": 0.43,
        "mean_gap": 0.25,
        "p10_gap": 0.15,
        "max_off_line_activation": 0.38,
        "mean_component_reduction_fraction": 0.30,
        "mean_removed_raw_foreground_fraction": 0.04,
        "mean_added_image_fraction": 0.01,
    }
    fragmented = dict(accepted, mean_component_reduction_fraction=0.10)

    assert all(acceptance_checks(baseline, accepted).values())
    assert not all(acceptance_checks(baseline, fragmented).values())


def test_choose_candidate_rejects_harmful_cleanup_and_keeps_best_eligible() -> None:
    baseline = {
        "name": "raw_day61",
        "mean_support": 0.44,
        "mean_gap": 0.25,
        "p10_gap": 0.15,
        "max_off_line_activation": 0.39,
        "score": 0.25,
    }
    eligible = dict(
        baseline,
        name="open3_close5_area_0001",
        score=0.27,
        mean_component_reduction_fraction=0.30,
        mean_removed_raw_foreground_fraction=0.04,
        mean_added_image_fraction=0.01,
    )
    harmful = dict(
        eligible,
        name="overcleaned",
        score=0.40,
        mean_support=0.38,
        mean_component_reduction_fraction=0.80,
    )

    chosen, checks = choose_candidate(baseline, [harmful, eligible])

    assert chosen["name"] == "open3_close5_area_0001"
    assert all(checks.values())


def test_choose_candidate_retains_raw_when_no_cleanup_passes() -> None:
    baseline = {
        "name": "raw_day61",
        "mean_support": 0.44,
        "mean_gap": 0.25,
        "p10_gap": 0.15,
        "max_off_line_activation": 0.39,
        "score": 0.25,
    }
    failing = dict(
        baseline,
        name="weak_cleanup",
        mean_component_reduction_fraction=0.05,
        mean_removed_raw_foreground_fraction=0.01,
        mean_added_image_fraction=0.0,
    )

    chosen, checks = choose_candidate(baseline, [failing])

    assert chosen["name"] == "raw_day61"
    assert checks == {"no_cleanup_candidate_passed": False}


def test_evaluate_configurations_reads_real_pairs_and_reports_both_metric_families(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "image"
    label_dir = tmp_path / "label"
    image_dir.mkdir()
    label_dir.mkdir()
    image = np.full((40, 60, 3), (70, 80, 100), dtype=np.uint8)
    cv2.rectangle(image, (25, 2), (35, 37), (20, 170, 25), -1)
    line = np.zeros((40, 60), dtype=np.uint8)
    cv2.line(line, (30, 38), (30, 2), 255, 1)
    assert cv2.imwrite(str(image_dir / "7.jpg"), image)
    assert cv2.imwrite(str(label_dir / "7.jpg"), line)
    configs = [
        {
            "name": "raw_day61",
            "order": None,
            "open_kernel": 3,
            "close_kernel": 5,
            "min_area_fraction": 0.0,
        },
        {
            "name": "open3_close5",
            "order": "open_close",
            "open_kernel": 3,
            "close_kernel": 5,
            "min_area_fraction": 0.0,
        },
    ]

    summaries, rows = evaluate_configurations(
        image_dir, label_dir, ["7"], configs
    )

    assert set(summaries) == {"raw_day61", "open3_close5"}
    assert len(rows) == 2
    assert {row["configuration"] for row in rows} == {
        "raw_day61",
        "open3_close5",
    }
    assert "line_neighborhood_support" in rows[0]
    assert "raw_component_count" in rows[0]
    assert "mean_band_support" in rows[0]
    assert "longest_supported_run_fraction" in rows[0]


def test_perspective_filter_keeps_small_top_region_but_rejects_same_size_at_bottom() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 8), (13, 11), 255, -1)
    cv2.rectangle(mask, (70, 88), (73, 91), 255, -1)

    filtered = filter_components_perspective(
        mask,
        base_area_fraction=0.002,
        top_scale=0.25,
        exponent=1.0,
    )

    assert filtered[9, 11] == 255
    assert filtered[89, 71] == 0


def test_vertical_metrics_reward_continuous_support_across_row_bands() -> None:
    line = np.zeros((80, 60), dtype=np.uint8)
    cv2.line(line, (30, 75), (30, 4), 255, 1)
    continuous = np.zeros_like(line)
    cv2.rectangle(continuous, (27, 2), (33, 77), 255, -1)
    fragmented = continuous.copy()
    fragmented[20:50] = 0

    good = vertical_line_support_metrics(continuous, line, bands=8, dilation_radius=4)
    bad = vertical_line_support_metrics(fragmented, line, bands=8, dilation_radius=4)

    assert good["mean_band_support"] > bad["mean_band_support"]
    assert good["supported_band_fraction"] > bad["supported_band_fraction"]
    assert good["longest_supported_run_fraction"] > bad["longest_supported_run_fraction"]


def test_tall_closing_kernel_connects_vertical_gap_without_bridging_horizontal_gap() -> None:
    mask = np.zeros((40, 50), dtype=np.uint8)
    cv2.rectangle(mask, (8, 4), (10, 10), 255, -1)
    cv2.rectangle(mask, (8, 14), (10, 20), 255, -1)
    cv2.rectangle(mask, (25, 28), (31, 30), 255, -1)
    cv2.rectangle(mask, (35, 28), (41, 30), 255, -1)

    closed = apply_morphology(
        mask,
        open_kernel=3,
        close_kernel=(3, 7),
        order="close",
    )

    assert closed[12, 9] == 255
    assert closed[29, 33] == 0


def test_clean_candidate_mask_supports_perspective_area_configuration() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 8), (13, 11), 255, -1)
    cv2.rectangle(mask, (70, 88), (73, 91), 255, -1)
    config = {
        "name": "perspective",
        "order": None,
        "open_kernel": 3,
        "close_kernel": 5,
        "min_area_fraction": 0.002,
        "perspective_top_scale": 0.25,
        "perspective_exponent": 1.0,
    }

    cleaned = clean_candidate_mask(mask, config)

    assert cleaned[9, 11] == 255
    assert cleaned[89, 71] == 0


def test_cross_validation_folds_are_deterministic_disjoint_and_balanced() -> None:
    stems = [str(index) for index in range(1250)]

    first = split_cv_folds(stems, folds=5)
    second = split_cv_folds(stems, folds=5)

    assert first == second
    assert [len(fold) for fold in first] == [250] * 5
    assert set().union(*map(set, first)) == set(stems)
    assert sum(len(set(a) & set(b)) for a in first for b in first if a is not b) == 0


def test_v2_gate_requires_less_deletion_geometry_readiness_and_v1_quality() -> None:
    v1 = {
        "mean_support": 0.44,
        "mean_gap": 0.266,
        "p10_gap": 0.176,
        "max_off_line_activation": 0.397,
        "mean_component_reduction_fraction": 0.85,
        "mean_cleaned_component_count": 43.0,
        "mean_removed_raw_foreground_fraction": 0.050,
        "mean_added_image_fraction": 0.009,
        "mean_vertical_band_support": 0.44,
        "p10_image_vertical_band_support": 0.20,
        "mean_supported_band_fraction": 0.80,
        "mean_longest_supported_run_fraction": 0.75,
    }
    improved = dict(
        v1,
        mean_support=0.445,
        mean_removed_raw_foreground_fraction=0.042,
        mean_vertical_band_support=0.45,
        p10_image_vertical_band_support=0.21,
    )
    noisy = dict(
        improved,
        mean_cleaned_component_count=90.0,
        mean_component_reduction_fraction=0.70,
    )

    assert all(v2_acceptance_checks(v1, improved).values())
    assert not all(v2_acceptance_checks(v1, noisy).values())


def test_v2_selection_requires_candidate_to_pass_at_least_four_of_five_folds() -> None:
    v1 = {
        "name": "v1_fixed_area",
        "score": 0.30,
        "mean_support": 0.44,
        "mean_gap": 0.266,
        "p10_gap": 0.176,
        "max_off_line_activation": 0.397,
        "mean_component_reduction_fraction": 0.85,
        "mean_cleaned_component_count": 43.0,
        "mean_removed_raw_foreground_fraction": 0.050,
        "mean_added_image_fraction": 0.009,
        "mean_vertical_band_support": 0.44,
        "p10_image_vertical_band_support": 0.20,
        "mean_supported_band_fraction": 0.80,
        "mean_longest_supported_run_fraction": 0.75,
    }
    stable = dict(
        v1,
        name="perspective_stable",
        score=0.302,
        mean_removed_raw_foreground_fraction=0.042,
        mean_vertical_band_support=0.45,
        p10_image_vertical_band_support=0.21,
    )
    unstable_good = dict(stable, name="perspective_unstable", score=0.40)
    unstable_bad = dict(
        unstable_good,
        mean_support=0.40,
        mean_component_reduction_fraction=0.60,
    )
    fold_summaries = []
    for index in range(5):
        fold_summaries.append(
            {
                "v1_fixed_area": v1,
                "perspective_stable": stable,
                "perspective_unstable": unstable_good if index < 3 else unstable_bad,
            }
        )

    selected, pass_counts = select_v2_from_fold_summaries(
        fold_summaries,
        baseline_name="v1_fixed_area",
        candidate_names=["perspective_stable", "perspective_unstable"],
    )

    assert selected == "perspective_stable"
    assert pass_counts == {"perspective_stable": 5, "perspective_unstable": 3}
