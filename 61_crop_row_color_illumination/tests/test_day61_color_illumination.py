from pathlib import Path
import json
import sys

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import day61_color_illumination as lesson

from day61_color_illumination import (
    compute_line_proxy_metrics,
    deterministic_experiment_split,
    normalized_exg,
    proxy_robustness_score,
    segment_exg_fixed,
    segment_exg_otsu,
    segment_hsv,
    segment_lab,
    select_representative_samples,
)


def test_gray_world_api_exists_before_behavior_checks() -> None:
    assert hasattr(lesson, "gray_world_balance")
    assert hasattr(lesson, "segment_grayworld_hsv")


def test_gray_world_keeps_neutral_gray_and_reduces_channel_cast() -> None:
    neutral = np.full((12, 16, 3), 100, dtype=np.uint8)
    cast = np.full((12, 16, 3), (160, 100, 60), dtype=np.uint8)

    neutral_result = lesson.gray_world_balance(neutral)
    corrected = lesson.gray_world_balance(cast)

    assert np.array_equal(neutral_result, neutral)
    before_spread = np.ptp(cast.reshape(-1, 3).mean(axis=0))
    after_spread = np.ptp(corrected.reshape(-1, 3).mean(axis=0))
    assert corrected.dtype == np.uint8
    assert after_spread < before_spread


def test_grayworld_hsv_returns_binary_mask_without_mutating_input() -> None:
    image = np.full((20, 30, 3), (120, 80, 50), dtype=np.uint8)
    image[:, :10] = (20, 150, 20)
    original = image.copy()

    mask = lesson.segment_grayworld_hsv(image)

    assert np.array_equal(image, original)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_validation_manifest_reader_uses_declared_item_ids(tmp_path: Path) -> None:
    assert hasattr(lesson, "read_manifest_stems")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"item_id": "3", "role": "validation_development"},
        {"item_id": "9", "role": "validation_development"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert lesson.read_manifest_stems(manifest, "validation_development") == ["3", "9"]


def test_normalized_exg_uses_float_and_scores_green_highest() -> None:
    image = np.array([[[0, 200, 0], [80, 80, 80], [30, 30, 150]]], dtype=np.uint8)

    exg = normalized_exg(image)

    assert exg.dtype == np.float32
    assert exg[0, 0] > exg[0, 1] > exg[0, 2]


def test_segmentation_masks_are_binary_and_keep_shape() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :15] = (0, 180, 0)
    image[:, 15:] = (60, 70, 110)

    masks = [
        segment_hsv(image),
        segment_lab(image),
        segment_exg_fixed(image),
        segment_exg_otsu(image),
    ]

    for mask in masks:
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})
        assert mask[:, :15].mean() > mask[:, 15:].mean()


def test_fixed_exg_does_not_force_neutral_image_into_two_classes() -> None:
    image = np.full((40, 60, 3), 100, dtype=np.uint8)
    image[10:20, 10:20] = (20, 160, 20)

    fixed = segment_exg_fixed(image)

    assert fixed[12:18, 12:18].mean() == 255
    assert (fixed > 0).mean() < 0.10


def test_line_proxy_metrics_reward_candidate_support_near_annotation() -> None:
    line = np.zeros((60, 60), dtype=np.uint8)
    cv2.line(line, (30, 55), (30, 5), 255, 1)
    on_line = np.zeros_like(line)
    cv2.rectangle(on_line, (26, 3), (34, 57), 255, -1)
    off_line = np.zeros_like(line)
    cv2.rectangle(off_line, (2, 3), (10, 57), 255, -1)

    good = compute_line_proxy_metrics(on_line, line, dilation_radius=5)
    bad = compute_line_proxy_metrics(off_line, line, dilation_radius=5)

    assert good["line_neighborhood_support"] > bad["line_neighborhood_support"]
    assert good["off_line_activation"] < bad["off_line_activation"]


def test_representative_selection_is_unique_and_covers_named_conditions() -> None:
    rows = [
        {
            "stem": str(index),
            "mean_v": float(index),
            "shadow_fraction": float(index % 5) / 5,
            "contrast_v": float(20 - index),
            "mean_s": float((index * 7) % 23),
        }
        for index in range(1, 31)
    ]

    selected = select_representative_samples(rows, count=8)

    assert len(selected) == 8
    assert len({row["stem"] for row in selected}) == 8
    assert {row["selection_reason"] for row in selected} >= {
        "darkest",
        "brightest",
        "most_shadow",
        "highest_contrast",
    }


def test_experiment_split_is_deterministic_disjoint_and_keeps_hard_cases_in_review() -> None:
    stems = [str(index) for index in range(1, 201)]
    hard = ["7", "19", "88"]

    first = deterministic_experiment_split(
        stems, hard_stems=hard, calibration_count=24, review_count=30, verification_count=40
    )
    second = deterministic_experiment_split(
        stems, hard_stems=hard, calibration_count=24, review_count=30, verification_count=40
    )

    assert first == second
    assert set(hard).issubset(first["review"])
    assert not (set(first["calibration"]) & set(first["review"]))
    assert not (set(first["calibration"]) & set(first["diagnostic"]))
    assert not (set(first["calibration"]) & set(first["verification"]))
    assert not (set(first["review"]) & set(first["diagnostic"]))
    assert not (set(first["review"]) & set(first["verification"]))
    assert not (set(first["diagnostic"]) & set(first["verification"]))


def test_proxy_score_penalizes_off_line_activation_and_bad_tail() -> None:
    good = proxy_robustness_score([0.42, 0.35], [0.10, 0.12])
    bad = proxy_robustness_score([0.55, 0.20], [0.45, 0.18])

    assert good["score"] > bad["score"]
    assert good["p10_gap"] > bad["p10_gap"]
