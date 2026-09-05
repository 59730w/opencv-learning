from __future__ import annotations

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from day63_crop_row_geometry import (  # noqa: E402
    DAY62_FROZEN_CONFIG,
    GeometryConfig,
    apply_normalized_roi,
    candidate_acceptance_checks,
    central_label_geometry,
    evaluate_prediction,
    extract_multiscale_geometry_features,
    fit_extra_trees_geometry,
    geometry_prediction_from_regression,
    hough_geometry,
    line_points_norm,
    perspective_matrices,
    perspective_support_geometry,
    run_day63_study,
    run_day63_v2_study,
    split_geometry_folds,
    summarize_metric_rows,
    select_train_only_candidate,
    v2_acceptance_checks,
)
import day63_crop_row_geometry as day63  # noqa: E402


def synthetic_row_mask(
    *,
    height: int = 240,
    width: int = 320,
    near_x_norm: float = 0.62,
    far_x_norm: float = 0.48,
    add_distractor: bool = False,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    near = (round(near_x_norm * (width - 1)), round(0.90 * (height - 1)))
    far = (round(far_x_norm * (width - 1)), round(0.40 * (height - 1)))
    cv2.line(mask, far, near, 255, 9)
    if add_distractor:
        cv2.line(mask, (20, height - 1), (120, round(0.40 * (height - 1))), 255, 12)
        mask[170:190, 250:290] = 255
    return mask


def synthetic_multirow_mask(*, broken: bool = False) -> np.ndarray:
    height, width = 240, 320
    mask = np.zeros((height, width), dtype=np.uint8)
    vanishing = np.array([0.52, 0.08])
    anchor_y = 0.40
    for anchor_x in (0.12, 0.31, 0.50, 0.69, 0.88):
        for y_norm in np.linspace(0.22, 0.92, 80):
            if broken and 0.54 < y_norm < 0.62:
                continue
            scale = (y_norm - vanishing[1]) / (anchor_y - vanishing[1])
            x_norm = vanishing[0] + (anchor_x - vanishing[0]) * scale
            x = round(x_norm * (width - 1))
            y = round(y_norm * (height - 1))
            if 0 <= x < width:
                cv2.circle(mask, (x, y), 4, 255, -1)
    return mask


def test_day62_input_configuration_is_frozen_v2() -> None:
    assert DAY62_FROZEN_CONFIG == {
        "name": "directional_close5x7_perspective40",
        "order": "open_close",
        "open_kernel": 3,
        "close_kernel": [5, 7],
        "min_area_fraction": 0.0002,
        "perspective_top_scale": 0.4,
        "perspective_exponent": 1.0,
    }


def test_apply_normalized_roi_is_resolution_independent() -> None:
    for shape in ((100, 200), (200, 400)):
        mask = np.full(shape, 255, dtype=np.uint8)
        roi = apply_normalized_roi(mask, top_y_norm=0.30)
        assert np.all(roi[: round(0.30 * (shape[0] - 1))] == 0)
        assert roi[-1].mean() == 255


def test_apply_normalized_roi_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="binary"):
        apply_normalized_roi(np.zeros((20, 20, 3), dtype=np.uint8), 0.3)
    with pytest.raises(ValueError, match="top_y_norm"):
        apply_normalized_roi(np.zeros((20, 20), dtype=np.uint8), 1.0)


def test_line_points_norm_uses_declared_near_and_far_rows() -> None:
    points = line_points_norm(near_x_px=159.5, far_x_px=79.75, width=320)
    assert points[0] == pytest.approx((0.5, 0.90))
    assert points[1] == pytest.approx((0.25, 0.40))


def test_perspective_support_search_recovers_slanted_central_row() -> None:
    mask = synthetic_row_mask(add_distractor=True)
    prediction = perspective_support_geometry(
        mask,
        GeometryConfig(
            name="synthetic",
            blur_sigma_x=5.0,
            center_penalty=0.8,
            angle_penalty=0.05,
        ),
    )
    assert prediction.status == "valid"
    assert prediction.near_x_norm == pytest.approx(0.62, abs=0.025)
    assert prediction.far_x_norm == pytest.approx(0.48, abs=0.025)
    assert 0.0 <= prediction.confidence <= 1.0


def test_empty_mask_is_rejected_instead_of_forcing_a_line() -> None:
    prediction = perspective_support_geometry(
        np.zeros((240, 320), dtype=np.uint8), GeometryConfig(name="empty")
    )
    assert prediction.status == "reject"
    assert prediction.near_x_norm is None
    assert prediction.far_x_norm is None
    assert prediction.confidence == 0.0


def test_broken_but_supported_row_is_not_downgraded_only_by_zero_p20() -> None:
    mask = np.zeros((240, 320), dtype=np.uint8)
    for y in range(96, 220, 30):
        cv2.line(mask, (160, y), (160, min(y + 9, 219)), 255, 9)
    prediction = perspective_support_geometry(
        mask,
        GeometryConfig(
            name="broken-row",
            center_penalty=1.5,
            angle_penalty=0.2,
        ),
    )
    assert prediction.mean_support >= 0.08
    assert prediction.p20_support < 0.03
    assert prediction.status == "valid"


def test_hough_baseline_returns_geometry_on_clean_line() -> None:
    prediction = hough_geometry(synthetic_row_mask())
    assert prediction.status != "reject"
    assert prediction.near_x_norm == pytest.approx(0.62, abs=0.04)


def test_central_label_geometry_chooses_nearest_bottom_row() -> None:
    label = np.zeros((240, 320), dtype=np.uint8)
    cv2.line(label, (20, 239), (145, 80), 255, 4)
    cv2.line(label, (190, 239), (160, 80), 255, 4)
    cv2.line(label, (310, 239), (175, 80), 255, 4)
    geometry = central_label_geometry(label)
    assert geometry.near_x_norm == pytest.approx(0.59, abs=0.04)
    assert geometry.status == "reference"


def test_evaluate_prediction_respects_position_and_heading_sign() -> None:
    reference = central_label_geometry(synthetic_row_mask())
    prediction = perspective_support_geometry(
        synthetic_row_mask(), GeometryConfig(name="metric", center_penalty=0.5)
    )
    metrics = evaluate_prediction(prediction, reference)
    assert metrics["bottom_position_abs_error_norm"] < 0.03
    assert metrics["heading_abs_error_deg"] < 2.0
    assert metrics["within_both_absolute_thresholds"] is True


def test_candidate_gate_requires_all_five_folds_and_absolute_thresholds() -> None:
    passing = [
        {
            "valid_fraction": 0.90,
            "bottom_position_mae_norm": 0.04,
            "heading_mae_deg": 4.0,
        }
        for _ in range(5)
    ]
    checks = candidate_acceptance_checks(passing)
    assert all(checks.values())

    passing[-1]["heading_mae_deg"] = 5.01
    assert candidate_acceptance_checks(passing)["all_five_folds_pass"] is False


def test_train_selection_never_uses_confirmation_metrics() -> None:
    folds = {
        "weaker": [
            {"valid_fraction": 0.9, "bottom_position_mae_norm": 0.045, "heading_mae_deg": 4.5}
            for _ in range(5)
        ],
        "stronger": [
            {"valid_fraction": 0.95, "bottom_position_mae_norm": 0.035, "heading_mae_deg": 3.8}
            for _ in range(5)
        ],
    }
    selected = select_train_only_candidate(folds)
    assert selected == "stronger"


def test_geometry_folds_are_deterministic_disjoint_and_balanced() -> None:
    stems = [str(index) for index in range(1250)]
    first = split_geometry_folds(stems)
    second = split_geometry_folds(list(reversed(stems)))
    assert first == second
    assert [len(fold) for fold in first] == [250] * 5
    assert set().union(*(set(fold) for fold in first)) == set(stems)


def test_metric_summary_keeps_tail_and_failure_evidence() -> None:
    rows = [
        {
            "status": "valid",
            "is_available": True,
            "bottom_position_abs_error_norm": 0.01,
            "heading_abs_error_deg": 1.0,
            "within_both_absolute_thresholds": True,
            "runtime_ms": 10.0,
        },
        {
            "status": "reject",
            "is_available": False,
            "bottom_position_abs_error_norm": None,
            "heading_abs_error_deg": None,
            "within_both_absolute_thresholds": False,
            "runtime_ms": 12.0,
        },
        {
            "status": "degraded",
            "is_available": True,
            "bottom_position_abs_error_norm": 0.40,
            "heading_abs_error_deg": 25.0,
            "within_both_absolute_thresholds": False,
            "runtime_ms": 14.0,
        },
    ]
    summary = summarize_metric_rows(rows)
    assert summary["valid_fraction"] == pytest.approx(1 / 3)
    assert summary["bottom_position_mae_norm"] == pytest.approx(0.01)
    assert summary["position_p90_norm"] == pytest.approx(0.01)
    assert summary["both_threshold_fraction_all"] == pytest.approx(1 / 3)
    assert summary["runtime_median_ms"] == pytest.approx(12.0)


def test_geometry_outside_central_half_is_degraded_not_forced_valid() -> None:
    mask = synthetic_row_mask(near_x_norm=0.18, far_x_norm=0.40)
    prediction = perspective_support_geometry(
        mask,
        GeometryConfig(
            name="edge-row",
            center_penalty=0.0,
            angle_penalty=0.0,
        ),
    )
    assert prediction.near_x_norm < 0.25
    assert prediction.status == "degraded"


def test_perspective_demo_has_inverse_mapping() -> None:
    matrix, inverse = perspective_matrices(width=320, height=240)
    point = np.array([[[160.0, 180.0]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(point, matrix)
    restored = cv2.perspectiveTransform(warped, inverse)
    assert restored == pytest.approx(point, abs=1e-3)


def test_tiny_study_preserves_evidence_boundaries_and_writes_outputs(tmp_path: Path) -> None:
    def make_split(name: str, count: int, role: str) -> tuple[Path, Path]:
        root = tmp_path / name
        (root / "image").mkdir(parents=True)
        (root / "label").mkdir()
        manifest = tmp_path / f"{name}.jsonl"
        records = []
        for index in range(count):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            label = synthetic_row_mask(
                height=120,
                width=160,
                near_x_norm=0.50 + 0.01 * ((index % 3) - 1),
                far_x_norm=0.50,
            )
            image[label > 0] = (0, 180, 0)
            cv2.imwrite(str(root / "image" / f"{index}.jpg"), image)
            cv2.imwrite(str(root / "label" / f"{index}.jpg"), label)
            records.append(
                json.dumps(
                    {"item_id": str(index), "role": role}, ensure_ascii=False
                )
            )
        manifest.write_text("\n".join(records), encoding="utf-8")
        return root, manifest

    train_root, train_manifest = make_split("train", 10, "train_development")
    validation_root, validation_manifest = make_split(
        "validation", 5, "validation_development"
    )
    output = tmp_path / "output"
    result = run_day63_study(
        train_root=train_root,
        train_manifest=train_manifest,
        validation_root=validation_root,
        validation_manifest=validation_manifest,
        output_dir=output,
        comparison_count=3,
    )
    assert result["marker"] == "DAY63_LESSON_COMPLETE"
    assert result["selection_data"] == "CRDLD train-development only"
    assert result["confirmation_is_untouched"] is False
    assert result["same_source_internal_benchmark_accessed"] is False
    assert result["frozen_external_accessed"] is False
    assert (output / "day63_results.json").is_file()
    assert (output / "geometry_metrics.csv").is_file()
    assert (output / "geometry_contact_sheet.jpg").is_file()


def test_multiscale_features_have_fixed_shape_and_resolution_transfer() -> None:
    small = synthetic_row_mask(height=240, width=320)
    large = synthetic_row_mask(height=480, width=640)
    small_features = extract_multiscale_geometry_features(small)
    large_features = extract_multiscale_geometry_features(large)
    assert small_features.shape == (320,)
    assert large_features.shape == (320,)
    assert np.mean(np.abs(small_features - large_features)) < 0.04


def test_extra_trees_geometry_is_deterministic_and_reports_uncertainty() -> None:
    rng = np.random.default_rng(63)
    features = rng.normal(size=(24, 12)).astype(np.float32)
    targets = np.column_stack(
        (0.5 + 0.04 * features[:, 0], 0.5 + 0.03 * features[:, 1])
    )
    first = fit_extra_trees_geometry(
        features,
        targets,
        n_estimators=24,
        max_depth=8,
        min_samples_leaf=2,
    )
    second = fit_extra_trees_geometry(
        features,
        targets,
        n_estimators=24,
        max_depth=8,
        min_samples_leaf=2,
    )
    mean_a, uncertainty_a = first.predict_with_uncertainty(features[:4])
    mean_b, uncertainty_b = second.predict_with_uncertainty(features[:4])
    assert mean_a == pytest.approx(mean_b)
    assert uncertainty_a == pytest.approx(uncertainty_b)
    assert np.all(uncertainty_a >= 0)


def test_v2_gate_requires_foldwise_tail_and_dual_threshold_improvement() -> None:
    v1 = [
        {
            "valid_fraction": 0.86,
            "bottom_position_mae_norm": 0.030,
            "heading_mae_deg": 3.6,
            "heading_p90_deg": 7.5,
            "both_threshold_fraction_all": 0.68,
        }
        for _ in range(5)
    ]
    v2 = [
        {
            "valid_fraction": 0.90,
            "bottom_position_mae_norm": 0.025,
            "heading_mae_deg": 2.2,
            "heading_p90_deg": 4.5,
            "both_threshold_fraction_all": 0.84,
        }
        for _ in range(5)
    ]
    assert all(v2_acceptance_checks(v1, v2).values())
    v2[-1]["both_threshold_fraction_all"] = 0.75
    assert v2_acceptance_checks(v1, v2)["all_folds_dual_gain_at_least_0_10"] is False


def test_regression_prediction_keeps_uncertainty_and_support_explainable() -> None:
    mask = synthetic_row_mask()
    prediction = geometry_prediction_from_regression(
        mask,
        endpoints=np.array([0.62, 0.48]),
        uncertainty=0.02,
        method="extra_trees_test",
    )
    assert prediction.status == "valid"
    assert prediction.near_x_norm == pytest.approx(0.62)
    assert prediction.heading_deg is not None
    assert prediction.mean_support > 0.1
    assert 0 <= prediction.confidence <= 1
    assert "uncertainty" in prediction.reason


def test_tiny_v2_study_writes_model_and_preserves_frozen_tests(tmp_path: Path) -> None:
    def make_split(name: str, count: int, role: str) -> tuple[Path, Path]:
        root = tmp_path / name
        (root / "image").mkdir(parents=True)
        (root / "label").mkdir()
        manifest = tmp_path / f"{name}.jsonl"
        records = []
        for index in range(count):
            near = 0.46 + 0.01 * (index % 5)
            label = synthetic_row_mask(
                height=120, width=160, near_x_norm=near, far_x_norm=0.50
            )
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[label > 0] = (0, 180, 0)
            cv2.imwrite(str(root / "image" / f"{index}.jpg"), image)
            cv2.imwrite(str(root / "label" / f"{index}.jpg"), label)
            records.append(json.dumps({"item_id": str(index), "role": role}))
        manifest.write_text("\n".join(records), encoding="utf-8")
        return root, manifest

    train_root, train_manifest = make_split("train_v2", 10, "train_development")
    val_root, val_manifest = make_split(
        "validation_v2", 5, "validation_development"
    )
    output = tmp_path / "v2_output"
    result = run_day63_v2_study(
        train_root=train_root,
        train_manifest=train_manifest,
        validation_root=val_root,
        validation_manifest=val_manifest,
        output_dir=output,
        comparison_count=3,
        candidate_configs=[
            {
                "name": "tiny_extra",
                "n_estimators": 12,
                "max_depth": 6,
                "min_samples_leaf": 1,
                "max_features": 1.0,
            }
        ],
    )
    assert result["marker"] == "DAY63_V2_LESSON_COMPLETE"
    assert result["same_source_internal_benchmark_accessed"] is False
    assert result["frozen_external_accessed"] is False
    assert result["untouched_confirmation_available"] is False
    assert (output / "day63_results_v2.json").is_file()
    assert (output / "day63_geometry_v2.joblib").is_file()
    assert (output / "geometry_metrics_v2.csv").is_file()


def test_multiline_label_extraction_recovers_all_ordered_rows() -> None:
    assert hasattr(day63, "extract_multirow_geometry")
    prediction = day63.extract_multirow_geometry(
        synthetic_multirow_mask(), label_mode=True
    )

    assert len(prediction.rows) == 5
    assert [row.far_x_norm for row in prediction.rows] == sorted(
        row.far_x_norm for row in prediction.rows
    )


def test_multiline_detector_keeps_broken_rows_as_distinct_hypotheses() -> None:
    prediction = day63.extract_multirow_geometry(
        synthetic_multirow_mask(broken=True), label_mode=False
    )

    assert len(prediction.rows) == 5
    assert prediction.status in {"valid", "degraded"}


def test_multiline_detector_uses_camera_prior_when_lower_mask_edges_bias_vp() -> None:
    mask = synthetic_multirow_mask(broken=True)
    cv2.line(mask, (5, 230), (145, 95), 255, 14)
    cv2.line(mask, (315, 230), (175, 105), 255, 14)

    prediction = day63.extract_multirow_geometry(mask, label_mode=False)

    assert len(prediction.rows) >= 4
    assert prediction.vanishing_point_norm is not None
    assert prediction.vanishing_point_norm[1] <= 0.08


def test_ordered_matching_penalizes_duplicate_and_missing_rows() -> None:
    CropRowLine = day63.CropRowLine
    reference = tuple(
        CropRowLine(x, x, 1.0, 4) for x in (0.2, 0.4, 0.6, 0.8)
    )
    predicted = tuple(
        CropRowLine(x, x, 0.8, 4) for x in (0.2, 0.21, 0.6)
    )

    metrics = day63.match_ordered_crop_rows(predicted, reference)

    assert metrics["matched_count"] == 2
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1 / 2)


def test_global_lattice_merges_duplicates_and_completes_supported_gap() -> None:
    CropRowLine = day63.CropRowLine
    candidates = tuple(
        CropRowLine(x, 0.5 + (x - 0.5) * 2.0, confidence, 5)
        for x, confidence in (
            (0.10, 0.8),
            (0.30, 0.9),
            (0.32, 0.4),
            (0.50, 0.9),
            (0.90, 0.8),
        )
    )

    assert hasattr(day63, "regularize_crop_row_lattice")
    rows = day63.regularize_crop_row_lattice(candidates, (0.5, 0.0))

    assert len(rows) == 5
    assert [row.far_x_norm for row in rows] == pytest.approx(
        [0.10, 0.30, 0.50, 0.70, 0.90], abs=0.025
    )
    observed = min(rows, key=lambda row: abs(row.far_x_norm - 0.30))
    assert observed.near_x_norm == pytest.approx(0.10, abs=0.04)
    inferred = min(rows, key=lambda row: abs(row.far_x_norm - 0.70))
    assert inferred.support_band_count == 0


def test_corridor_uses_adjacent_rows_and_rejects_a_central_crop_row() -> None:
    CropRowLine = day63.CropRowLine
    clear = tuple(
        CropRowLine(x, x, 1.0, 4) for x in (0.18, 0.38, 0.64, 0.86)
    )
    corridor = day63.derive_image_corridor(clear)
    assert corridor is not None
    assert corridor.near_x_norm == pytest.approx(0.51)

    blocked = tuple(
        CropRowLine(x, x, 1.0, 4) for x in (0.2, 0.5, 0.8)
    )
    assert day63.derive_image_corridor(blocked) is None

    unsupported_boundary = tuple(
        CropRowLine(x, x, 0.8, bands) for x, bands in ((0.38, 0), (0.64, 5))
    )
    assert day63.derive_image_corridor(unsupported_boundary) is None
    assert day63._corridor_indices(unsupported_boundary) is None


def test_tiny_multirow_revision_writes_scoped_results(tmp_path: Path) -> None:
    root = tmp_path / "multirow"
    (root / "image").mkdir(parents=True)
    (root / "label").mkdir()
    records = []
    for index in range(4):
        label = synthetic_multirow_mask(broken=index % 2 == 1)
        image = np.zeros((*label.shape, 3), dtype=np.uint8)
        image[label > 0] = (0, 180, 0)
        cv2.imwrite(str(root / "image" / f"{index}.jpg"), image)
        cv2.imwrite(str(root / "label" / f"{index}.jpg"), label)
        records.append(str(index))
    train_manifest = tmp_path / "train_manifest.jsonl"
    validation_manifest = tmp_path / "validation_manifest.jsonl"
    train_manifest.write_text(
        "\n".join(json.dumps({"item_id": stem, "role": "train_development"}) for stem in records),
        encoding="utf-8",
    )
    validation_manifest.write_text(
        "\n".join(json.dumps({"item_id": stem, "role": "validation_development"}) for stem in records),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    assert hasattr(day63, "run_day63_multirow_study")
    result = day63.run_day63_multirow_study(
        train_root=root,
        train_manifest=train_manifest,
        validation_root=root,
        validation_manifest=validation_manifest,
        output_dir=output,
        comparison_count=2,
    )

    assert result["marker"] == "DAY63_MULTIROW_REVISION_COMPLETE"
    assert result["label_identity_scope"] == "derived_ordered_matching"
    assert result["frozen_external_accessed"] is False
    assert (output / "day63_results_multirow.json").is_file()
    assert (output / "geometry_metrics_multirow.csv").is_file()
    assert (output / "geometry_comparison_multirow.jpg").is_file()


def test_centerline_tensor_keeps_rgb_day62_mask_and_label_shapes() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    mask = synthetic_multirow_mask()
    label = mask.copy()

    features, target = day63.prepare_centerline_tensor(
        image, mask, label, resolution=96
    )

    assert features.shape == (4, 96, 96)
    assert target.shape == (1, 96, 96)
    assert features.dtype == np.uint8
    assert target.dtype == np.uint8
    assert set(np.unique(target)).issubset({0, 1})
    assert features[3].max() == 255


def test_centerline_heatmap_decoder_recovers_ordered_rows_and_heading() -> None:
    mask = synthetic_multirow_mask()
    probability = cv2.resize(
        mask.astype(np.float32) / 255.0,
        (128, 128),
        interpolation=cv2.INTER_AREA,
    )

    prediction = day63.decode_centerline_heatmap(
        probability,
        peak_height=0.20,
        peak_prominence=0.05,
        peak_distance_norm=0.025,
    )

    assert len(prediction.rows) == 5
    assert [row.far_x_norm for row in prediction.rows] == sorted(
        row.far_x_norm for row in prediction.rows
    )
    assert max(abs(row.heading_deg) for row in prediction.rows) > 1.0


def test_crossfit_folds_are_deterministic_disjoint_and_cover_every_item() -> None:
    stems = [f"sample-{index}" for index in range(17)]

    first = day63.split_crossfit_folds(stems, fold_count=3)
    second = day63.split_crossfit_folds(stems, fold_count=3)

    assert first == second
    assert sorted(stem for fold in first for stem in fold) == sorted(stems)
    assert not (set(first[0]) & set(first[1]))


def test_tiny_row_unet_preserves_spatial_resolution() -> None:
    torch = pytest.importorskip("torch")
    model = day63.TinyRowUNet(base_channels=4)

    output = model(torch.zeros((2, 4, 64, 64), dtype=torch.float32))

    assert tuple(output.shape) == (2, 1, 64, 64)


def test_resnet18_row_unet_accepts_rgb_plus_day62_mask() -> None:
    torch = pytest.importorskip("torch")
    model = day63.ResNet18RowUNet()
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros((1, 4, 64, 64), dtype=torch.float32))

    assert tuple(output.shape) == (1, 1, 64, 64)
