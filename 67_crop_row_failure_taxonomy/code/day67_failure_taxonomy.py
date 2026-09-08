"""Day67 event-based failure taxonomy for the frozen Day66 development pilot.

The module separates machine-observable state-machine triggers from reviewed
visual scene labels.  It never estimates real-video safety accuracy because no
framewise corridor-validity ground truth exists.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import cv2
import numpy as np


ALLOWED_STATES = {"valid", "candidate", "degraded", "reject"}
FROZEN_ROLE = "frozen_same_source_holdout"


class ProtocolError(ValueError):
    """Raised when an input violates the frozen Day67 evidence protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def normalize_system_trigger(reason: str) -> str:
    """Map Day66 reason text to a trigger, not an inferred visual cause."""

    exact = {
        "both supported adjacent crop-row boundaries are required": "insufficient_adjacent_boundaries",
        "central crop row occupies the camera corridor; no drivable center emitted": "central_row_guard",
        "different corridor awaits switch confirmation": "corridor_switch_guard",
        "Day63 rejected rows; no downstream corridor measurement promoted": "upstream_row_rejection",
        "no live crop-row tracks remain": "no_live_tracks",
        "corridor candidate awaits consecutive confirmation": "initial_confirmation_guard",
        "known corridor is recovering after interruption": "recovery_confirmation_guard",
        "same corridor geometry awaits reidentification confirmation": "reidentification_guard",
        "same observed boundary identities remain confirmed": "stable_valid_output",
        "new corridor confirmed before identity switch": "new_corridor_valid_output",
        "initial corridor confirmed across consecutive frames": "initial_valid_output",
        "same corridor geometry reidentified with new row tracks": "reidentified_valid_output",
    }
    if reason in exact:
        return exact[reason]
    return "unmapped_reason"


def _pair(record: dict[str, Any]) -> tuple[int, ...] | None:
    diagnostics = record.get("diagnostics") or {}
    pair = diagnostics.get("active_or_selected_track_ids")
    if pair is None:
        return None
    return tuple(int(value) for value in pair)


def _event_key(record: dict[str, Any]) -> tuple[str, str, tuple[int, ...] | None]:
    state = str(record["pilot_state"])
    trigger = normalize_system_trigger(str(record.get("reason") or ""))
    identity_sensitive = state in {"valid", "candidate"} or trigger == "reidentification_guard"
    return (state, trigger, _pair(record) if identity_sensitive else None)


def build_failure_events(
    records: Iterable[dict[str, Any]], context_frames: int = 12
) -> list[dict[str, Any]]:
    """Collapse frame bursts into episode-local, auditable event units."""

    if context_frames < 0:
        raise ProtocolError("context_frames must be non-negative")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episode_order: list[str] = []
    for item in records:
        episode = str(item.get("episode") or "")
        if not episode:
            raise ProtocolError("record is missing episode")
        if episode not in grouped:
            episode_order.append(episode)
        grouped[episode].append(item)

    events: list[dict[str, Any]] = []
    for episode in episode_order:
        episode_records = grouped[episode]
        indices = [int(item["frame_index"]) for item in episode_records]
        if any(right <= left for left, right in zip(indices, indices[1:])):
            raise ProtocolError(f"frame indices must be strictly increasing in {episode}")
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise ProtocolError(f"records must be contiguous in {episode}")
        episode_max = indices[-1]

        start = 0
        while start < len(episode_records):
            key = _event_key(episode_records[start])
            end = start
            while end + 1 < len(episode_records) and _event_key(episode_records[end + 1]) == key:
                end += 1
            first = episode_records[start]
            last = episode_records[end]
            state, trigger, pair = key
            start_frame = int(first["frame_index"])
            end_frame = int(last["frame_index"])
            event_id = f"{episode}__{start_frame:06d}_{end_frame:06d}__{state}__{trigger}"
            events.append({
                "event_id": event_id,
                "episode": episode,
                "role": str(first.get("role") or ""),
                "pilot_state": state,
                "system_trigger": trigger,
                "reason": str(first.get("reason") or ""),
                "track_pair": list(pair) if pair is not None else None,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": end_frame - start_frame + 1,
                "representative_frame": (start_frame + end_frame) // 2,
                "context_start_frame": max(indices[0], start_frame - context_frames),
                "context_end_frame": min(episode_max, end_frame + context_frames),
                "navigation_available": bool(first.get("navigation_available")),
            })
            start = end + 1
    return events


def aggregate_event_inventory(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    event_list = list(events)

    def grouped(field: str) -> dict[str, dict[str, int]]:
        buckets: dict[str, dict[str, Any]] = {}
        for event in event_list:
            key = str(event[field])
            bucket = buckets.setdefault(key, {"event_count": 0, "frame_count": 0, "episodes": set()})
            bucket["event_count"] += 1
            bucket["frame_count"] += int(event["frame_count"])
            bucket["episodes"].add(event["episode"])
        return {
            key: {
                "event_count": value["event_count"],
                "frame_count": value["frame_count"],
                "episode_count": len(value["episodes"]),
            }
            for key, value in sorted(buckets.items())
        }

    return {
        "event_count": len(event_list),
        "frame_count": sum(int(event["frame_count"]) for event in event_list),
        "episode_count": len({event["episode"] for event in event_list}),
        "by_pilot_state": grouped("pilot_state"),
        "by_system_trigger": grouped("system_trigger"),
    }


def select_review_events(
    events: Iterable[dict[str, Any]], per_stratum: int = 3, seed: int = 6701
) -> list[dict[str, Any]]:
    """Keep all valid events and sample non-valid strata across episodes."""

    if per_stratum < 1:
        raise ProtocolError("per_stratum must be positive")
    event_list = list(events)
    selected: dict[str, dict[str, Any]] = {
        item["event_id"]: item for item in event_list if item["pilot_state"] == "valid"
    }
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in event_list:
        if item["pilot_state"] != "valid":
            strata[(item["role"], item["pilot_state"], item["system_trigger"])].append(item)

    rng = random.Random(seed)
    for stratum in sorted(strata):
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in strata[stratum]:
            by_episode[item["episode"]].append(item)
        episode_names = sorted(by_episode)
        rng.shuffle(episode_names)
        choices: list[dict[str, Any]] = []
        for episode in episode_names:
            candidates = sorted(by_episode[episode], key=lambda value: value["event_id"])
            choices.append(candidates[rng.randrange(len(candidates))])
            if len(choices) == per_stratum:
                break
        if len(choices) < per_stratum:
            remaining = [
                item for item in sorted(strata[stratum], key=lambda value: value["event_id"])
                if item["event_id"] not in {choice["event_id"] for choice in choices}
            ]
            rng.shuffle(remaining)
            choices.extend(remaining[: per_stratum - len(choices)])
        for item in choices:
            selected[item["event_id"]] = item
    return sorted(selected.values(), key=lambda item: (item["episode"], item["start_frame"]))


def _allocate_stratified_sample(
    sizes: dict[tuple[str, str, str], int], target: int, minimum: int
) -> dict[tuple[str, str, str], int]:
    """Allocate a fixed budget with small-stratum census and proportional remainder."""

    total = sum(sizes.values())
    if target < 1 or minimum < 1:
        raise ProtocolError("sampling target and minimum must be positive")
    target = min(target, total)
    allocation = {key: min(size, minimum) for key, size in sizes.items()}
    if sum(allocation.values()) > target:
        raise ProtocolError("target is too small for the minimum per stratum")
    remaining = target - sum(allocation.values())
    while remaining:
        eligible = [key for key in sizes if allocation[key] < sizes[key]]
        residual_total = sum(sizes[key] - allocation[key] for key in eligible)
        quotas = {key: remaining * (sizes[key] - allocation[key]) / residual_total for key in eligible}
        added = 0
        for key in sorted(eligible):
            count = min(sizes[key] - allocation[key], int(quotas[key]))
            allocation[key] += count
            added += count
        remaining -= added
        if remaining:
            ranked = sorted(
                (key for key in eligible if allocation[key] < sizes[key]),
                key=lambda key: (-(quotas[key] - int(quotas[key])), key),
            )
            take = min(remaining, len(ranked))
            for key in ranked[:take]:
                allocation[key] += 1
            remaining -= take
    return allocation


def select_review_events_v2(
    events: Iterable[dict[str, Any]], *, target_nonvalid: int = 240,
    minimum_per_stratum: int = 10, seed: int = 6702,
) -> list[dict[str, Any]]:
    """Census valid events and draw an auditable weighted non-valid sample."""

    event_list = list(events)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in event_list:
        if item["pilot_state"] != "valid":
            strata[(item["role"], item["pilot_state"], item["system_trigger"])].append(item)
    allocation = _allocate_stratified_sample(
        {key: len(value) for key, value in strata.items()}, target_nonvalid, minimum_per_stratum
    )
    valid_count = sum(item["pilot_state"] == "valid" for item in event_list)
    selected: list[dict[str, Any]] = []
    for item in event_list:
        if item["pilot_state"] == "valid":
            enriched = dict(item)
            enriched.update({"stratum_id": "valid_census", "stratum_population_event_count": valid_count,
                             "stratum_sample_event_count": valid_count, "inclusion_probability": 1.0,
                             "design_weight": 1.0})
            selected.append(enriched)
    rng = random.Random(seed)
    for key in sorted(strata):
        population = sorted(strata[key], key=lambda value: value["event_id"])
        sample_count = allocation[key]
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in population:
            by_episode[item["episode"]].append(item)
        for values in by_episode.values():
            rng.shuffle(values)
        episode_order = sorted(by_episode)
        rng.shuffle(episode_order)
        choices: list[dict[str, Any]] = []
        while len(choices) < sample_count:
            progressed = False
            for episode in episode_order:
                if by_episode[episode] and len(choices) < sample_count:
                    choices.append(by_episode[episode].pop())
                    progressed = True
            if not progressed:
                break
            rng.shuffle(episode_order)
        probability = sample_count / len(population)
        for item in choices:
            enriched = dict(item)
            enriched.update({"stratum_id": "|".join(key),
                             "stratum_population_event_count": len(population),
                             "stratum_sample_event_count": sample_count,
                             "inclusion_probability": probability,
                             "design_weight": 1.0 / probability})
            selected.append(enriched)
    return sorted(selected, key=lambda item: (item["episode"], item["start_frame"]))


def select_review_events_v3(
    events: Iterable[dict[str, Any]], *, target_nonvalid: int = 240,
    minimum_per_stratum: int = 10, seed: int = 6703,
) -> list[dict[str, Any]]:
    """Use SRSWOR inside each predeclared stratum so N_h/n_h is valid."""

    event_list = list(events)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in event_list:
        if item["pilot_state"] != "valid":
            strata[(item["role"], item["pilot_state"], item["system_trigger"])].append(item)
    allocation = _allocate_stratified_sample(
        {key: len(value) for key, value in strata.items()}, target_nonvalid, minimum_per_stratum
    )
    valid_count = sum(item["pilot_state"] == "valid" for item in event_list)
    selected: list[dict[str, Any]] = []
    for item in event_list:
        if item["pilot_state"] == "valid":
            enriched = dict(item)
            enriched.update({"stratum_id": "valid_census", "stratum_population_event_count": valid_count,
                             "stratum_sample_event_count": valid_count, "inclusion_probability": 1.0,
                             "design_weight": 1.0, "sampling_method": "census"})
            selected.append(enriched)
    rng = random.Random(seed)
    for key in sorted(strata):
        population = sorted(strata[key], key=lambda value: value["event_id"])
        sample_count = allocation[key]
        probability = sample_count / len(population)
        for item in rng.sample(population, sample_count):
            enriched = dict(item)
            enriched.update({"stratum_id": "|".join(key),
                             "stratum_population_event_count": len(population),
                             "stratum_sample_event_count": sample_count,
                             "inclusion_probability": probability,
                             "design_weight": 1.0 / probability,
                             "sampling_method": "srswor_within_stratum"})
            selected.append(enriched)
    return sorted(selected, key=lambda item: (item["episode"], item["start_frame"]))


def make_annotation_template(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create deliberately uncommitted labels for a two-pass visual review."""

    return [{
        "event_id": item["event_id"],
        "reviewability": "ungradable",
        "primary_visual_label": "ungradable",
        "secondary_visual_labels": [],
        "navigation_assessment": "ungradable",
        "review_passes": [],
        "notes": "",
    } for item in events]


def resolve_review_sources(
    events: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
    day66_output: Path,
    protocol: dict[str, Any],
) -> dict[str, dict[str, Path]]:
    """Resolve only development sources; reject frozen/unknown names before I/O."""

    manifest_by_name = {item["episode"]: item for item in manifest.get("episodes", [])}
    allowed_roles = set(protocol["allowed_roles"])
    sources: dict[str, dict[str, Path]] = {}
    for event in events:
        episode = str(event["episode"])
        if episode not in manifest_by_name:
            raise ProtocolError(f"unknown episode requested for review: {episode}")
        entry = manifest_by_name[episode]
        if entry.get("role") == FROZEN_ROLE:
            raise ProtocolError(f"frozen episode requested for review: {episode}")
        if entry.get("role") not in allowed_roles:
            raise ProtocolError(f"disallowed episode requested for review: {episode}")
        sources[episode] = {
            "raw_path": Path(entry["video_path"]),
            "overlay_path": day66_output / f"{episode}_day66_overlay.mp4",
        }
    return sources


def _read_video_frame(capture: cv2.VideoCapture, frame_index: int, source: Path) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ProtocolError(f"cannot decode frame {frame_index} from {source}")
    return frame


def render_review_contact_sheets(
    events: Iterable[dict[str, Any]],
    sources: dict[str, dict[str, Path]],
    output_dir: Path,
    *,
    columns: int = 4,
    rows: int = 4,
) -> list[Path]:
    """Render representative raw/overlay pairs; source resolution is prevalidated."""

    if columns < 1 or rows < 1:
        raise ProtocolError("contact-sheet rows and columns must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    event_list = list(events)
    panels: list[np.ndarray] = []
    captures: dict[tuple[str, str], cv2.VideoCapture] = {}
    try:
        for review_index, event in enumerate(event_list):
            episode = event["episode"]
            if episode not in sources:
                raise ProtocolError(f"review source was not resolved: {episode}")
            source = sources[episode]
            frames = []
            for kind in ("raw", "overlay"):
                path = source[f"{kind}_path"]
                key = (episode, kind)
                if key not in captures:
                    if not path.is_file():
                        raise ProtocolError(f"review video does not exist: {path}")
                    captures[key] = cv2.VideoCapture(str(path))
                    if not captures[key].isOpened():
                        raise ProtocolError(f"cannot open review video: {path}")
                frame = _read_video_frame(captures[key], int(event["representative_frame"]), path)
                frames.append(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA))
            body = np.hstack(frames)
            header = np.full((48, body.shape[1], 3), 24, dtype=np.uint8)
            label = (
                f"R{review_index:03d} {episode} f={event['representative_frame']} "
                f"{event['pilot_state']} {event['system_trigger']}"
            )
            cv2.putText(header, label[:92], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(header, f"event={event['event_id'][-72:]}", (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 220, 255), 1, cv2.LINE_AA)
            panels.append(np.vstack([header, body]))
    finally:
        for capture in captures.values():
            capture.release()

    page_size = rows * columns
    sheet_paths: list[Path] = []
    blank = np.full((228, 640, 3), 16, dtype=np.uint8)
    for page_start in range(0, len(panels), page_size):
        page = panels[page_start: page_start + page_size]
        page += [blank.copy() for _ in range(page_size - len(page))]
        row_images = [np.hstack(page[index:index + columns]) for index in range(0, page_size, columns)]
        sheet = np.vstack(row_images)
        path = output_dir / f"day67_review_sheet_{page_start // page_size:03d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ProtocolError(f"cannot write contact sheet: {path}")
        sheet_paths.append(path)
    return sheet_paths


def render_review_context_clips(
    events: Iterable[dict[str, Any]],
    sources: dict[str, dict[str, Path]],
    output_dir: Path,
    *,
    fps: float = 10.0,
) -> list[Path]:
    """Write frame-aligned raw/overlay context clips for selected audit events."""

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    for event in events:
        episode = event["episode"]
        if episode not in sources:
            raise ProtocolError(f"review source was not resolved: {episode}")
        raw_path = sources[episode]["raw_path"]
        overlay_path = sources[episode]["overlay_path"]
        raw = cv2.VideoCapture(str(raw_path))
        overlay = cv2.VideoCapture(str(overlay_path))
        if not raw.isOpened() or not overlay.isOpened():
            raw.release()
            overlay.release()
            raise ProtocolError(f"cannot open context sources for {event['event_id']}")
        output_path = output_dir / f"{event['event_id']}_context.mp4"
        writer: cv2.VideoWriter | None = None
        written = 0
        try:
            start = int(event["context_start_frame"])
            end = int(event["context_end_frame"])
            raw.set(cv2.CAP_PROP_POS_FRAMES, start)
            overlay.set(cv2.CAP_PROP_POS_FRAMES, start)
            for frame_index in range(start, end + 1):
                raw_ok, raw_frame = raw.read()
                overlay_ok, overlay_frame = overlay.read()
                if not raw_ok or not overlay_ok or raw_frame is None or overlay_frame is None:
                    raise ProtocolError(f"context decode failed at {episode} frame {frame_index}")
                if raw_frame.shape[:2] != overlay_frame.shape[:2]:
                    overlay_frame = cv2.resize(
                        overlay_frame,
                        (raw_frame.shape[1], raw_frame.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                combined = np.hstack([raw_frame, overlay_frame])
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (combined.shape[1], combined.shape[0]),
                    )
                    if not writer.isOpened():
                        raise ProtocolError(f"cannot create context clip: {output_path}")
                writer.write(combined)
                written += 1
        finally:
            raw.release()
            overlay.release()
            if writer is not None:
                writer.release()
        expected = int(event["context_end_frame"]) - int(event["context_start_frame"]) + 1
        if written != expected:
            raise ProtocolError(f"context clip frame count mismatch: {event['event_id']}")
        clip_paths.append(output_path)
    return clip_paths


def validate_navigation_invariant(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    navigation_fields = (
        "corridor_centerline_points_norm",
        "lateral_offset_norm",
        "heading_error_deg",
    )
    violations: list[dict[str, Any]] = []
    for item in records:
        state = item.get("pilot_state")
        valid_contract = (
            state == "valid"
            and item.get("status") == "valid"
            and item.get("navigation_available") is True
            and all(item.get(field) is not None for field in navigation_fields)
        )
        nonvalid_contract = (
            state != "valid"
            and item.get("navigation_available") is False
            and all(item.get(field) is None for field in navigation_fields)
        )
        if not (valid_contract or nonvalid_contract):
            violations.append({"episode": item.get("episode"), "frame_index": item.get("frame_index")})
    return violations


def validate_annotation(
    event: dict[str, Any], annotation: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    if annotation.get("event_id") != event.get("event_id"):
        raise ProtocolError("annotation event_id does not match event")
    reviewability = annotation.get("reviewability")
    if reviewability not in protocol["reviewability"]:
        raise ProtocolError("unsupported reviewability")
    label = annotation.get("primary_visual_label")
    if label not in protocol["visual_labels"]:
        raise ProtocolError("unsupported primary visual label")
    secondary = annotation.get("secondary_visual_labels") or []
    if any(item not in protocol["visual_labels"] for item in secondary):
        raise ProtocolError("unsupported secondary visual label")
    navigation = annotation.get("navigation_assessment")
    if navigation not in protocol["navigation_assessments"]:
        raise ProtocolError("unsupported navigation assessment")
    if reviewability == "ungradable" and (label != "ungradable" or secondary):
        raise ProtocolError("ungradable review cannot invent a scene cause")
    passes = annotation.get("review_passes") or []
    if len(passes) < 2:
        raise ProtocolError("two review passes are required")
    output = dict(annotation)
    output.update({
        "episode": event["episode"],
        "pilot_state": event["pilot_state"],
        "system_trigger": event["system_trigger"],
        "frame_count": event["frame_count"],
    })
    return output


def materialize_review_annotations(
    review_events: Iterable[dict[str, Any]],
    decisions: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand recorded visual decisions without deriving labels from triggers."""

    passes = decisions.get("review_passes") or []
    valid_default = decisions.get("valid_default")
    nonvalid_default = decisions.get("reviewed_nonvalid_default")
    overrides = decisions.get("overrides_by_review_index") or {}
    overrides_by_event_id = decisions.get("overrides_by_event_id") or {}
    batch_overrides: dict[str, dict[str, Any]] = {}
    for batch in decisions.get("batch_overrides_by_review_indices") or []:
        for raw_index in batch.get("review_indices") or []:
            batch_overrides[str(int(raw_index))] = batch["decision"]
    nonvalid_count = sum(item["pilot_state"] != "valid" for item in review_events)
    nonvalid_attested = (
        decisions.get("review_attestation") == "all_manifest_events_reviewed_twice"
        and int(decisions.get("reviewed_nonvalid_event_count", -1)) == nonvalid_count
    )
    output: list[dict[str, Any]] = []
    for index, event in enumerate(review_events):
        decision = overrides_by_event_id.get(
            event["event_id"], overrides.get(str(index), batch_overrides.get(str(index)))
        )
        if decision is None:
            if event["pilot_state"] != "valid" and nonvalid_default is not None and nonvalid_attested:
                decision = nonvalid_default
            elif event["pilot_state"] != "valid" or valid_default is None:
                raise ProtocolError(
                    f"explicit review decision is required for non-valid review index {index}"
                )
            else:
                decision = valid_default
        annotation = dict(decision)
        annotation.update({"event_id": event["event_id"], "review_passes": list(passes)})
        output.append(validate_annotation(event, annotation, protocol))
    unknown_indices = set(overrides) - {str(index) for index in range(len(output))}
    if unknown_indices:
        raise ProtocolError(f"review decisions contain unknown indices: {sorted(unknown_indices)}")
    unknown_event_ids = set(overrides_by_event_id) - {item["event_id"] for item in output}
    if unknown_event_ids:
        raise ProtocolError(f"review decisions contain unknown event IDs: {sorted(unknown_event_ids)}")
    unknown_batch_indices = set(batch_overrides) - {str(index) for index in range(len(output))}
    if unknown_batch_indices:
        raise ProtocolError(f"batch review decisions contain unknown indices: {sorted(unknown_batch_indices)}")
    return output


def aggregate_review(
    events: Iterable[dict[str, Any]], annotations: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    event_map = {item["event_id"]: item for item in events}
    weighted_design = any("design_weight" in item for item in event_map.values())
    by_label: dict[str, dict[str, Any]] = {}
    reviewed_ids: set[str] = set()
    for annotation in annotations:
        event_id = annotation["event_id"]
        if event_id not in event_map:
            raise ProtocolError(f"annotation references unknown event: {event_id}")
        if event_id in reviewed_ids:
            raise ProtocolError(f"duplicate annotation: {event_id}")
        reviewed_ids.add(event_id)
        event = event_map[event_id]
        label = annotation["primary_visual_label"]
        bucket = by_label.setdefault(label, {
            "event_count": 0, "frame_count": 0, "episodes": set(),
            "estimated_population_event_count": 0.0,
        })
        bucket["event_count"] += 1
        bucket["frame_count"] += int(event["frame_count"])
        bucket["episodes"].add(event["episode"])
        bucket["estimated_population_event_count"] += float(event.get("design_weight", 1.0))
    normalized = {
        label: ({
            "event_count": bucket["event_count"],
            "frame_count": bucket["frame_count"],
            "episode_count": len(bucket["episodes"]),
        } | ({"estimated_population_event_count": bucket["estimated_population_event_count"]}
             if weighted_design else {}))
        for label, bucket in sorted(by_label.items())
    }
    return {
        "reviewed_event_count": len(reviewed_ids),
        "reviewed_event_span_frame_count": sum(event_map[event_id]["frame_count"] for event_id in reviewed_ids),
        "reviewed_episode_count": len({event_map[event_id]["episode"] for event_id in reviewed_ids}),
        "by_primary_visual_label": normalized,
    }


def choose_day68_priority(summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    rules = protocol["day68_decision"]
    reviewed = int(summary.get("reviewed_event_count", 0))
    if reviewed < int(rules["minimum_reviewed_events"]):
        return {"status": "BLOCKED", "selected_cause": None, "rationale": "too few reviewed events for the preregistered gate"}
    actionable_denominator = sum(
        float(counts.get("estimated_population_event_count", counts["event_count"]))
        for label, counts in summary.get("by_primary_visual_label", {}).items()
        if protocol["visual_labels"].get(label, {}).get("actionable_day68")
    )
    if actionable_denominator == 0:
        return {
            "status": "BLOCKED",
            "selected_cause": None,
            "rationale": "no reviewed actionable failure events",
            "actionable_failure_event_denominator": 0,
            "actionable_failure_estimated_event_denominator": 0.0,
        }
    eligible = []
    for label, counts in summary.get("by_primary_visual_label", {}).items():
        definition = protocol["visual_labels"].get(label, {})
        estimated_count = float(counts.get("estimated_population_event_count", counts["event_count"]))
        fraction = estimated_count / actionable_denominator
        if (
            definition.get("actionable_day68")
            and counts["episode_count"] >= int(rules["minimum_episodes"])
            and fraction >= float(rules["minimum_event_fraction"])
        ):
            eligible.append((
                int(definition.get("severity", 0)),
                int(counts["episode_count"]),
                fraction,
                label,
            ))
    if not eligible:
        return {
            "status": "BLOCKED",
            "selected_cause": None,
            "rationale": "no actionable cause passed the cross-episode prevalence gate",
            "actionable_failure_event_denominator": actionable_denominator,
            "actionable_failure_estimated_event_denominator": actionable_denominator,
        }
    winner = max(eligible)
    return {
        "status": "PASS",
        "selected_cause": winner[3],
        "rationale": "selected by severity, then cross-episode support, then actionable-failure event prevalence; raw frame duration did not select the cause",
        "severity": winner[0],
        "episode_count": winner[1],
        "actionable_failure_event_fraction": winner[2],
        "actionable_failure_event_denominator": actionable_denominator,
        "actionable_failure_estimated_event_denominator": actionable_denominator,
    }


def build_acceptance_checks(
    inventory: dict[str, Any],
    review_events: Iterable[dict[str, Any]],
    annotations: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, bool]:
    review_list = list(review_events)
    annotation_list = list(annotations)
    annotation_ids = {item.get("event_id") for item in annotation_list}
    valid_ids = {item["event_id"] for item in review_list if item["pilot_state"] == "valid"}
    review_ids = [item["event_id"] for item in review_list]
    return {
        "development_frames_reconciled": inventory["frame_count"] == metadata["development_frame_count"],
        "development_episodes_reconciled": inventory["episode_count"] == metadata["development_episode_count"],
        "frozen_frames_accessed_is_zero": metadata["frozen_frames_accessed"] == 0,
        "event_ids_unique": len(review_ids) == len(set(review_ids)),
        "all_review_events_annotated": len(annotation_list) == len(review_list) and annotation_ids == set(review_ids),
        "all_valid_events_reviewed": valid_ids <= annotation_ids,
        "two_or_more_review_passes_recorded": all(
            len(item.get("review_passes") or []) >= 2 for item in annotation_list
        ),
    }


def finalize_review(
    review_events: Iterable[dict[str, Any]],
    decisions: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_list = list(review_events)
    annotations = materialize_review_annotations(event_list, decisions, protocol)
    summary = aggregate_review(event_list, annotations)
    decision = choose_day68_priority(summary, protocol)
    unsafe_ids = {
        item["event_id"] for item in annotations
        if item["pilot_state"] == "valid" and item["navigation_assessment"] == "suspected_unsafe"
    }
    event_map = {item["event_id"]: item for item in event_list}
    report = {
        "schema_version": int(protocol.get("schema_version", 1)),
        "marker": "DAY67_FAILURE_TAXONOMY_COMPLETE",
        "protocol_id": protocol.get("protocol_id", "test_protocol"),
        "evidence_status": protocol.get("evidence_status", "MODEL_ASSISTED_REVIEW_DEVELOPMENT_ONLY"),
        "review_summary": summary,
        "suspected_unsafe_valid_event_count": len(unsafe_ids),
        "suspected_unsafe_valid_frame_count": sum(event_map[event_id]["frame_count"] for event_id in unsafe_ids),
        "suspected_unsafe_valid_episode_count": len({event_map[event_id]["episode"] for event_id in unsafe_ids}),
        "day68_priority_decision": decision,
        "real_video_safety_gate": protocol.get(
            "real_video_safety_gate",
            "BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH",
        ),
        "review_claim_boundary": (
            "Visual labels are two-pass model-assisted development review, not "
            "independent human annotation or framewise corridor-validity ground truth."
        ),
    }
    return report, annotations


def load_and_validate_day66(
    output_dir: Path, result_path: Path, manifest_path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = load_json(result_path)
    manifest = load_json(manifest_path)
    if result.get("marker") != "DAY66_OFFLINE_PILOT_COMPLETE":
        raise ProtocolError("Day66 result marker is not complete")
    if bool(result.get("frozen_video_frames_accessed")):
        raise ProtocolError("Day66 reports frozen frames were accessed")
    manifest_episodes = {item["episode"]: item for item in manifest.get("episodes", [])}
    frozen = {name for name, item in manifest_episodes.items() if item.get("role") == FROZEN_ROLE}
    allowed_roles = set(protocol["allowed_roles"])
    allowed = {name for name, item in manifest_episodes.items() if item.get("role") in allowed_roles}

    jsonl_paths = sorted(output_dir.glob("*.jsonl"))
    if any(path.stem in frozen for path in jsonl_paths):
        raise ProtocolError("frozen episode JSONL is present in Day66 output")
    unknown = {path.stem for path in jsonl_paths} - allowed
    if unknown:
        raise ProtocolError(f"unknown or disallowed episode outputs: {sorted(unknown)}")
    if {path.stem for path in jsonl_paths} != allowed:
        missing = sorted(allowed - {path.stem for path in jsonl_paths})
        raise ProtocolError(f"Day66 development episode set does not exactly match manifest: {missing}")

    records: list[dict[str, Any]] = []
    episode_counts: dict[str, int] = {}
    for path in jsonl_paths:
        episode_records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("episode") != path.stem:
                    raise ProtocolError(f"episode mismatch in {path}")
                if item.get("role") not in allowed_roles:
                    raise ProtocolError(f"frozen or disallowed role in {path}")
                if item.get("pilot_state") not in ALLOWED_STATES:
                    raise ProtocolError(f"unsupported pilot state in {path}")
                episode_records.append(item)
        expected = int(manifest_episodes[path.stem]["video"]["decoded_frame_count"])
        if len(episode_records) != expected:
            raise ProtocolError(f"frame count mismatch for {path.stem}")
        episode_counts[path.stem] = len(episode_records)
        records.extend(episode_records)
    result_frames = result.get("aggregate", {}).get("frame_count", result.get("aggregate", {}).get("total_frames"))
    if result_frames is not None and int(result_frames) != len(records):
        raise ProtocolError("Day66 aggregate frame count does not match JSONL")
    if int(result.get("episode_count", len(episode_counts))) != len(episode_counts):
        raise ProtocolError("Day66 episode count does not match JSONL")
    violations = validate_navigation_invariant(records)
    if violations:
        raise ProtocolError(f"navigation invariant violations: {len(violations)}")
    return records, {
        "development_episode_count": len(episode_counts),
        "development_frame_count": len(records),
        "frozen_episode_count": len(frozen),
        "frozen_frames_accessed": 0,
        "day66_result_sha256": sha256_file(result_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day66-output", type=Path, required=True)
    parser.add_argument("--day66-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("day67_taxonomy.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-review-packet", action="store_true")
    parser.add_argument("--review-decisions", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    protocol_data = load_json(args.protocol)
    records, metadata = load_and_validate_day66(
        args.day66_output, args.day66_result, args.manifest, protocol_data
    )
    events = build_failure_events(records, int(protocol_data["context_frames"]))
    selected = select_review_events_v3(
        events,
        target_nonvalid=int(protocol_data["target_nonvalid_review_events"]),
        minimum_per_stratum=int(protocol_data["minimum_review_events_per_stratum"]),
        seed=int(protocol_data["random_seed"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "day67_events.jsonl").open("w", encoding="utf-8") as handle:
        for item in events:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_dir / "day67_review_manifest_v3.jsonl").open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    annotation_template = make_annotation_template(selected)
    with (args.output_dir / "day67_annotations_template_v3.jsonl").open("w", encoding="utf-8") as handle:
        for item in annotation_template:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    sheet_paths = sorted((args.output_dir / "review_sheets_v3").glob("day67_review_sheet_*.jpg"))
    if args.render_review_packet:
        manifest_data = load_json(args.manifest)
        sources = resolve_review_sources(selected, manifest_data, args.day66_output, protocol_data)
        sheet_paths = render_review_contact_sheets(
            selected, sources, args.output_dir / "review_sheets_v3"
        )
    report = {
        "schema_version": int(protocol_data.get("schema_version", 1)),
        "marker": "DAY67_REVIEW_PACKET_READY",
        "protocol_id": protocol_data["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "metadata": metadata,
        "event_count": len(events),
        "review_event_count": len(selected),
        "valid_event_count": sum(item["pilot_state"] == "valid" for item in events),
        "review_includes_all_valid_events": all(
            item["event_id"] in {chosen["event_id"] for chosen in selected}
            for item in events if item["pilot_state"] == "valid"
        ),
        "review_sheet_count": len(sheet_paths),
        "annotation_template_count": len(annotation_template),
        "evidence_status": protocol_data["evidence_status"],
        "real_video_safety_gate": protocol_data["real_video_safety_gate"],
    }
    (args.output_dir / "day67_packet_report_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.review_decisions is not None:
        decisions = load_json(args.review_decisions)
        manifest_path = args.output_dir / "day67_review_manifest_v3.jsonl"
        if decisions.get("review_manifest_sha256") != sha256_file(manifest_path):
            raise ProtocolError("review decisions do not match the frozen v2 review manifest")
        if int(decisions.get("review_event_count", -1)) != len(selected):
            raise ProtocolError("review decision count does not match the v2 manifest")
        final_report, annotations = finalize_review(selected, decisions, protocol_data)
        final_report["packet_metadata"] = metadata
        inventory = aggregate_event_inventory(events)
        final_report["full_inventory"] = inventory
        final_report["full_event_count"] = len(events)
        final_report["review_event_count"] = len(selected)
        final_report["all_valid_events_reviewed"] = report["review_includes_all_valid_events"]
        final_report["review_decisions_sha256"] = sha256_file(args.review_decisions)
        final_report["navigation_invariant_violations"] = 0
        final_report["acceptance_checks"] = build_acceptance_checks(
            inventory, selected, annotations, metadata
        )
        final_report["acceptance_checks"]["sampling_weights_reconcile_full_event_population"] = (
            abs(sum(float(item["design_weight"]) for item in selected) - len(events)) < 1e-6
        )
        final_report["acceptance_checks"]["review_manifest_hash_bound_to_decisions"] = (
            decisions["review_manifest_sha256"] == sha256_file(manifest_path)
        )
        final_report["day67_engineering_gate_passed"] = all(
            final_report["acceptance_checks"].values()
        )
        with (args.output_dir / "day67_annotations_v3.jsonl").open("w", encoding="utf-8") as handle:
            for item in annotations:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        (args.output_dir / "day67_results_v3.json").write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = final_report
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
