from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
DAY67_CODE = ROOT / "67_crop_row_failure_taxonomy" / "code"
if str(DAY67_CODE) not in sys.path:
    sys.path.insert(0, str(DAY67_CODE))

from day67_failure_taxonomy import (  # noqa: E402
    ProtocolError,
    aggregate_event_inventory,
    aggregate_review,
    build_failure_events,
    build_acceptance_checks,
    choose_day68_priority,
    finalize_review,
    load_and_validate_day66,
    make_annotation_template,
    materialize_review_annotations,
    normalize_system_trigger,
    render_review_contact_sheets,
    render_review_context_clips,
    resolve_review_sources,
    select_review_events,
    select_review_events_v2,
    select_review_events_v3,
    validate_annotation,
    validate_navigation_invariant,
)


def test_v2_sampling_is_seeded_capacity_aware_and_records_design_weights() -> None:
    events = []
    for index in range(20):
        events.append({
            "event_id": f"large_{index}", "episode": f"ep_{index % 5}",
            "role": "temporal_development", "pilot_state": "degraded",
            "system_trigger": "central_row_guard", "start_frame": index,
        })
    for index in range(3):
        events.append({
            "event_id": f"small_{index}", "episode": f"small_ep_{index}",
            "role": "shifted_development", "pilot_state": "reject",
            "system_trigger": "no_live_tracks", "start_frame": index,
        })
    events.append({
        "event_id": "valid_0", "episode": "ep_0", "role": "temporal_development",
        "pilot_state": "valid", "system_trigger": "stable_valid_output", "start_frame": 30,
    })
    first = select_review_events_v2(events, target_nonvalid=10, minimum_per_stratum=4, seed=6702)
    second = select_review_events_v2(events, target_nonvalid=10, minimum_per_stratum=4, seed=6702)
    assert [item["event_id"] for item in first] == [item["event_id"] for item in second]
    assert len(first) == 11
    assert next(item for item in first if item["event_id"] == "valid_0")["design_weight"] == 1.0
    small = [item for item in first if item["event_id"].startswith("small_")]
    assert len(small) == 3
    assert all(item["inclusion_probability"] == 1.0 for item in small)
    large = [item for item in first if item["event_id"].startswith("large_")]
    assert len(large) == 7
    assert all(item["stratum_population_event_count"] == 20 for item in large)
    assert all(item["stratum_sample_event_count"] == 7 for item in large)
    assert all(item["design_weight"] == pytest.approx(20 / 7) for item in large)


def test_v3_sampling_is_equal_probability_within_each_stratum() -> None:
    import random

    events = [{
        "event_id": f"event_{index:02d}",
        "episode": "large_episode" if index < 18 else f"small_episode_{index}",
        "role": "temporal_development", "pilot_state": "degraded",
        "system_trigger": "central_row_guard", "start_frame": index,
    } for index in range(20)]
    seed = 6703
    selected = select_review_events_v3(
        events, target_nonvalid=7, minimum_per_stratum=1, seed=seed
    )
    expected_ids = {
        item["event_id"] for item in random.Random(seed).sample(
            sorted(events, key=lambda item: item["event_id"]), 7
        )
    }
    assert {item["event_id"] for item in selected} == expected_ids
    assert all(item["inclusion_probability"] == pytest.approx(7 / 20) for item in selected)
    assert all(item["sampling_method"] == "srswor_within_stratum" for item in selected)


def test_weighted_review_drives_day68_fraction_without_inflating_episode_support() -> None:
    summary = {
        "reviewed_event_count": 6,
        "by_primary_visual_label": {
            "severe_occlusion": {
                "event_count": 2, "frame_count": 2, "episode_count": 2,
                "estimated_population_event_count": 20.0,
            },
            "central_row_occupancy": {
                "event_count": 4, "frame_count": 4, "episode_count": 3,
                "estimated_population_event_count": 5.0,
            },
        },
    }
    decision = choose_day68_priority(summary, protocol())
    assert decision["selected_cause"] == "severe_occlusion"
    assert decision["actionable_failure_estimated_event_denominator"] == pytest.approx(25.0)
    assert decision["actionable_failure_event_fraction"] == pytest.approx(0.8)


def record(
    frame: int,
    *,
    episode: str = "ep_a",
    role: str = "temporal_development",
    state: str = "degraded",
    reason: str = "both supported adjacent crop-row boundaries are required",
    pair: list[int] | None = None,
    navigation: bool = False,
) -> dict:
    return {
        "episode": episode,
        "frame_index": frame,
        "role": role,
        "pilot_state": state,
        "status": "valid" if navigation else ("reject" if state == "reject" else "degraded"),
        "navigation_available": navigation,
        "corridor_centerline_points_norm": [[0.5, 0.9], [0.5, 0.4]] if navigation else None,
        "lateral_offset_norm": 0.0 if navigation else None,
        "heading_error_deg": 0.0 if navigation else None,
        "reason": reason,
        "diagnostics": {"active_or_selected_track_ids": pair},
    }


def protocol() -> dict:
    return {
        "allowed_roles": ["temporal_development", "shifted_development"],
        "reviewability": ["reviewable", "ungradable"],
        "navigation_assessments": ["appropriate", "suspected_unsafe", "not_applicable", "ungradable"],
        "visual_labels": {
            "suspected_ood_valid": {"severity": 5, "actionable_day68": True},
            "central_row_occupancy": {"severity": 3, "actionable_day68": True},
            "missing_both_boundaries": {"severity": 3, "actionable_day68": True},
            "severe_occlusion": {"severity": 4, "actionable_day68": True},
            "ungradable": {"severity": 0, "actionable_day68": False},
            "no_failure_visible": {"severity": 0, "actionable_day68": False},
        },
        "day68_decision": {
            "minimum_reviewed_events": 3,
            "minimum_episodes": 2,
            "minimum_event_fraction": 0.10,
        },
    }


def test_normalizes_day66_reasons_without_claiming_visual_cause() -> None:
    assert normalize_system_trigger("both supported adjacent crop-row boundaries are required") == "insufficient_adjacent_boundaries"
    assert normalize_system_trigger("central crop row occupies the camera corridor; no drivable center emitted") == "central_row_guard"
    assert normalize_system_trigger("different corridor awaits switch confirmation") == "corridor_switch_guard"
    assert normalize_system_trigger("Day63 rejected rows; no downstream corridor measurement promoted") == "upstream_row_rejection"
    assert normalize_system_trigger("no live crop-row tracks remain") == "no_live_tracks"


def test_event_segmentation_uses_episode_state_trigger_and_pair() -> None:
    records = [
        record(0),
        record(1),
        record(2, state="candidate", reason="corridor candidate awaits consecutive confirmation", pair=[1, 2]),
        record(3, state="candidate", reason="corridor candidate awaits consecutive confirmation", pair=[1, 2]),
        record(4, state="candidate", reason="corridor candidate awaits consecutive confirmation", pair=[3, 4]),
    ]
    events = build_failure_events(records, context_frames=2)
    assert [(e["start_frame"], e["end_frame"], e["frame_count"]) for e in events] == [
        (0, 1, 2), (2, 3, 2), (4, 4, 1)
    ]
    assert events[0]["event_id"] == "ep_a__000000_000001__degraded__insufficient_adjacent_boundaries"
    assert events[0]["context_start_frame"] == 0
    assert events[0]["context_end_frame"] == 3


def test_ordinary_degraded_event_is_not_fragmented_by_track_churn() -> None:
    records = [record(0, pair=[1, 2]), record(1, pair=[3, 4]), record(2, pair=None)]
    events = build_failure_events(records)
    assert len(events) == 1
    assert events[0]["frame_count"] == 3


def test_switch_guard_burst_is_one_event_even_when_pending_pair_changes() -> None:
    reason = "different corridor awaits switch confirmation"
    records = [record(0, reason=reason, pair=[1, 2]), record(1, reason=reason, pair=[3, 4])]
    events = build_failure_events(records)
    assert len(events) == 1


def test_event_segmentation_rejects_gaps_and_duplicate_frames() -> None:
    with pytest.raises(ProtocolError, match="contiguous"):
        build_failure_events([record(0), record(2)])
    with pytest.raises(ProtocolError, match="strictly increasing"):
        build_failure_events([record(0), record(0)])


def test_selection_keeps_every_valid_event_and_is_seeded_and_episode_aware() -> None:
    records = []
    for ep in ("ep_a", "ep_b", "ep_c", "ep_d"):
        records.extend([
            record(0, episode=ep),
            record(1, episode=ep, state="valid", reason="same observed boundary identities remain confirmed", pair=[1, 2], navigation=True),
            record(2, episode=ep),
        ])
    events = build_failure_events(records)
    first = select_review_events(events, per_stratum=2, seed=6701)
    second = select_review_events(events, per_stratum=2, seed=6701)
    assert [e["event_id"] for e in first] == [e["event_id"] for e in second]
    valid_ids = {e["event_id"] for e in events if e["pilot_state"] == "valid"}
    assert valid_ids <= {e["event_id"] for e in first}
    nonvalid = [e for e in first if e["pilot_state"] != "valid"]
    assert len({e["episode"] for e in nonvalid}) >= 2


def test_annotation_contract_separates_system_trigger_and_scene_label() -> None:
    event = build_failure_events([record(0)])[0]
    annotation = validate_annotation(
        event,
        {
            "event_id": event["event_id"],
            "reviewability": "reviewable",
            "primary_visual_label": "severe_occlusion",
            "secondary_visual_labels": ["missing_both_boundaries"],
            "navigation_assessment": "not_applicable",
            "review_passes": ["pass_1", "pass_2"],
            "notes": "Leaves hide the near-field boundary.",
        },
        protocol(),
    )
    assert annotation["system_trigger"] == "insufficient_adjacent_boundaries"
    assert annotation["primary_visual_label"] == "severe_occlusion"


def test_ungradable_annotation_cannot_invent_a_scene_cause() -> None:
    event = build_failure_events([record(0)])[0]
    with pytest.raises(ProtocolError, match="ungradable"):
        validate_annotation(
            event,
            {
                "event_id": event["event_id"],
                "reviewability": "ungradable",
                "primary_visual_label": "severe_occlusion",
                "secondary_visual_labels": [],
                "navigation_assessment": "ungradable",
                "review_passes": ["pass_1", "pass_2"],
                "notes": "",
            },
            protocol(),
        )


def test_nonvalid_navigation_fields_are_always_empty() -> None:
    assert validate_navigation_invariant([record(0)]) == []
    leaking = record(0)
    leaking["corridor_centerline_points_norm"] = [[0.4, 0.9], [0.5, 0.4]]
    assert validate_navigation_invariant([leaking]) == [{"episode": "ep_a", "frame_index": 0}]


def test_review_aggregation_reports_events_frames_and_independent_episodes() -> None:
    events = build_failure_events([
        record(0, episode="ep_a"), record(1, episode="ep_a"),
        record(0, episode="ep_b"),
    ])
    annotations = []
    for event in events:
        annotations.append(validate_annotation(event, {
            "event_id": event["event_id"],
            "reviewability": "reviewable",
            "primary_visual_label": "missing_both_boundaries",
            "secondary_visual_labels": [],
            "navigation_assessment": "not_applicable",
            "review_passes": ["pass_1", "pass_2"],
            "notes": "",
        }, protocol()))
    summary = aggregate_review(events, annotations)
    label = summary["by_primary_visual_label"]["missing_both_boundaries"]
    assert label == {"event_count": 2, "frame_count": 3, "episode_count": 2}


def test_full_inventory_aggregation_keeps_event_frame_and_episode_counts_separate() -> None:
    events = build_failure_events([
        record(0, episode="ep_a"), record(1, episode="ep_a"),
        record(0, episode="ep_b"),
    ])
    summary = aggregate_event_inventory(events)
    assert summary["event_count"] == 2
    assert summary["frame_count"] == 3
    assert summary["episode_count"] == 2
    assert summary["by_system_trigger"]["insufficient_adjacent_boundaries"] == {
        "event_count": 2, "frame_count": 3, "episode_count": 2
    }


def test_day68_choice_requires_cross_episode_prevalence_and_actionability() -> None:
    summary = {
        "reviewed_event_count": 10,
        "by_primary_visual_label": {
            "suspected_ood_valid": {"event_count": 4, "frame_count": 100, "episode_count": 1},
            "central_row_occupancy": {"event_count": 3, "frame_count": 20, "episode_count": 3},
        },
    }
    decision = choose_day68_priority(summary, protocol())
    assert decision["status"] == "PASS"
    assert decision["selected_cause"] == "central_row_occupancy"
    assert "cross-episode" in decision["rationale"]


def test_day68_choice_blocks_when_no_cause_passes_preregistered_gate() -> None:
    summary = {
        "reviewed_event_count": 2,
        "by_primary_visual_label": {
            "suspected_ood_valid": {"event_count": 2, "frame_count": 200, "episode_count": 1},
        },
    }
    decision = choose_day68_priority(summary, protocol())
    assert decision["status"] == "BLOCKED"
    assert decision["selected_cause"] is None


def test_day68_prevalence_denominator_excludes_reviewed_no_failure_controls() -> None:
    summary = {
        "reviewed_event_count": 100,
        "by_primary_visual_label": {
            "no_failure_visible": {"event_count": 95, "frame_count": 95, "episode_count": 10},
            "severe_occlusion": {"event_count": 5, "frame_count": 20, "episode_count": 3},
        },
    }
    decision = choose_day68_priority(summary, protocol())
    assert decision["status"] == "PASS"
    assert decision["selected_cause"] == "severe_occlusion"
    assert decision["actionable_failure_event_denominator"] == 5


def test_loader_reconciles_result_manifest_and_rejects_frozen_role(tmp_path: Path) -> None:
    output = tmp_path / "day66"
    output.mkdir()
    (output / "ep_a.jsonl").write_text(json.dumps(record(0)) + "\n", encoding="utf-8")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "marker": "DAY66_OFFLINE_PILOT_COMPLETE",
        "episode_count": 1,
        "aggregate": {"frame_count": 1},
        "frozen_video_frames_accessed": 0,
    }), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [
        {"episode": "ep_a", "role": "temporal_development", "video": {"decoded_frame_count": 1}, "video_path": "a.mp4"},
        {"episode": "frozen", "role": "frozen_same_source_holdout", "video": {"decoded_frame_count": 1}, "video_path": "f.mp4"},
    ]}), encoding="utf-8")
    records, metadata = load_and_validate_day66(output, results, manifest, protocol())
    assert len(records) == 1
    assert metadata["frozen_episode_count"] == 1
    assert metadata["frozen_frames_accessed"] == 0

    (output / "frozen.jsonl").write_text(json.dumps(record(0, episode="frozen", role="frozen_same_source_holdout")) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="frozen"):
        load_and_validate_day66(output, results, manifest, protocol())


def test_review_source_resolution_cannot_open_frozen_or_unknown_episode(tmp_path: Path) -> None:
    manifest = {
        "episodes": [
            {"episode": "ep_a", "role": "temporal_development", "video_path": str(tmp_path / "ep_a.mp4")},
            {"episode": "frozen", "role": "frozen_same_source_holdout", "video_path": str(tmp_path / "frozen.mp4")},
        ]
    }
    sources = resolve_review_sources(
        [{"episode": "ep_a"}], manifest, tmp_path / "day66", protocol()
    )
    assert sources["ep_a"]["overlay_path"].name == "ep_a_day66_overlay.mp4"
    with pytest.raises(ProtocolError, match="frozen"):
        resolve_review_sources([{"episode": "frozen"}], manifest, tmp_path, protocol())
    with pytest.raises(ProtocolError, match="unknown"):
        resolve_review_sources([{"episode": "missing"}], manifest, tmp_path, protocol())


def test_annotation_template_is_conservative_and_complete() -> None:
    event = build_failure_events([record(0)])[0]
    template = make_annotation_template([event])
    assert template == [{
        "event_id": event["event_id"],
        "reviewability": "ungradable",
        "primary_visual_label": "ungradable",
        "secondary_visual_labels": [],
        "navigation_assessment": "ungradable",
        "review_passes": [],
        "notes": "",
    }]


def test_contact_sheet_renders_raw_and_overlay_without_touching_other_sources(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.mp4"
    overlay_path = tmp_path / "overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for path, color in ((raw_path, (0, 80, 0)), (overlay_path, (0, 0, 120))):
        writer = cv2.VideoWriter(str(path), fourcc, 5.0, (64, 36))
        for _ in range(2):
            writer.write(np.full((36, 64, 3), color, dtype=np.uint8))
        writer.release()
    event = build_failure_events([record(0), record(1)])[0]
    sheets = render_review_contact_sheets(
        [event],
        {"ep_a": {"raw_path": raw_path, "overlay_path": overlay_path}},
        tmp_path / "sheets",
        columns=1,
        rows=1,
    )
    assert len(sheets) == 1
    image = cv2.imread(str(sheets[0]))
    assert image is not None
    assert image.shape[1] > image.shape[0]


def test_materialized_review_requires_explicit_decision_for_every_nonvalid_event() -> None:
    events = build_failure_events([
        record(0, state="valid", reason="same observed boundary identities remain confirmed", pair=[1, 2], navigation=True),
        record(1),
    ])
    decisions = {
        "review_passes": ["contact_sheet_pass_1", "context_pass_2"],
        "valid_default": {
            "reviewability": "reviewable",
            "primary_visual_label": "no_failure_visible",
            "secondary_visual_labels": [],
            "navigation_assessment": "appropriate",
            "notes": "Corridor appears visually plausible in both passes.",
        },
        "overrides_by_review_index": {},
    }
    with pytest.raises(ProtocolError, match="explicit review decision"):
        materialize_review_annotations(events, decisions, protocol())
    decisions["overrides_by_review_index"]["1"] = {
        "reviewability": "reviewable",
        "primary_visual_label": "missing_both_boundaries",
        "secondary_visual_labels": [],
        "navigation_assessment": "not_applicable",
        "notes": "No defensible adjacent pair is visible.",
    }
    annotations = materialize_review_annotations(events, decisions, protocol())
    assert [item["primary_visual_label"] for item in annotations] == [
        "no_failure_visible", "missing_both_boundaries"
    ]


def test_context_clip_is_frame_aligned_raw_plus_overlay(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.mp4"
    overlay_path = tmp_path / "overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for path, color in ((raw_path, (0, 80, 0)), (overlay_path, (0, 0, 120))):
        writer = cv2.VideoWriter(str(path), fourcc, 5.0, (64, 36))
        for _ in range(3):
            writer.write(np.full((36, 64, 3), color, dtype=np.uint8))
        writer.release()
    event = build_failure_events([record(0), record(1), record(2)], context_frames=1)[0]
    clips = render_review_context_clips(
        [event],
        {"ep_a": {"raw_path": raw_path, "overlay_path": overlay_path}},
        tmp_path / "clips",
    )
    capture = cv2.VideoCapture(str(clips[0]))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 128
    capture.release()


def test_finalize_review_preserves_safety_block_and_counts_suspected_unsafe() -> None:
    events = build_failure_events([
        record(0, state="valid", reason="same observed boundary identities remain confirmed", pair=[1, 2], navigation=True),
        record(1),
    ])
    decisions = {
        "review_passes": ["pass_1", "pass_2"],
        "valid_default": {
            "reviewability": "reviewable",
            "primary_visual_label": "suspected_ood_valid",
            "secondary_visual_labels": [],
            "navigation_assessment": "suspected_unsafe",
            "notes": "OOD scene.",
        },
        "overrides_by_review_index": {
            "1": {
                "reviewability": "reviewable",
                "primary_visual_label": "missing_both_boundaries",
                "secondary_visual_labels": [],
                "navigation_assessment": "not_applicable",
                "notes": "No pair.",
            }
        },
    }
    report, annotations = finalize_review(events, decisions, protocol())
    assert len(annotations) == 2
    assert report["suspected_unsafe_valid_event_count"] == 1
    assert report["real_video_safety_gate"] == "BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH"


def test_acceptance_checks_require_reconciliation_all_valid_review_and_frozen_isolation() -> None:
    events = build_failure_events([
        record(0, state="valid", reason="same observed boundary identities remain confirmed", pair=[1, 2], navigation=True),
        record(1),
    ])
    annotations = [
        {"event_id": item["event_id"], "review_passes": ["pass_1", "pass_2"]}
        for item in events
    ]
    checks = build_acceptance_checks(
        aggregate_event_inventory(events),
        events,
        annotations,
        {"development_episode_count": 1, "development_frame_count": 2, "frozen_frames_accessed": 0},
    )
    assert all(checks.values())
    missing_valid = build_acceptance_checks(
        aggregate_event_inventory(events), events, annotations[1:],
        {"development_episode_count": 1, "development_frame_count": 2, "frozen_frames_accessed": 0},
    )
    assert missing_valid["all_valid_events_reviewed"] is False
