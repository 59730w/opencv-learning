# Day66 Offline Video Pilot Implementation Plan

**Status:** Executed and verified on 2026-09-07. The user reviewed Day66 v2 and then authorized README/Skill updates plus GitHub delivery.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Completed steps are recorded with checked boxes.

**Goal:** Build and execute a complete, reproducible Day66 offline video pilot using the frozen Day65 temporal configuration, producing continuous overlays, contract-shaped frame records, runtime evidence, and an evidence-bounded Chinese learning note.

**Architecture:** Add one Day66 orchestration module that imports the accepted Day65 predictor/tracker instead of duplicating perception logic. It verifies a tracked frozen-config artifact against the accepted Day65 result, processes only the manifest's development roles, projects diagnostic four-state output into the three-state navigation contract, renders a second decoding pass for continuous overlays, and writes checksummed per-episode and aggregate evidence. Frozen May 5 holdout videos are structurally excluded and remain untouched until Day69.

**Tech Stack:** Python 3.9, OpenCV 4.10, PyTorch 2.5.1 CUDA 12.1, pytest, JSON/JSONL, SHA-256.

## Global Constraints

- During implementation, work in `D:/opencv-learning` on the user-authorized current `main` checkout and do not commit or push before the review gate. That gate was later approved.
- Use only `temporal_development` and `shifted_development` CROW roles; never open, infer, render, tune, or audit `frozen_same_source_holdout`.
- The Day65 selected parameters and optical-flow setting are immutable during Day66.
- Improvement is allowed only if a preregistered Day66 acceptance check fails, and only against development evidence; every attempt must be retained and the frozen group remains inaccessible.
- Synthetic safety invariants are engineering evidence, not real-video safety accuracy.
- Large videos, JSONL records, contact sheets, and runtime traces stay under `D:/DL_code/data/crop_row_perception/day66_offline_video_pilot` and are not tracked by Git.
- The public navigation contract has `valid/degraded/reject`; `candidate` remains a diagnostic pilot state and maps to public `degraded` with navigation unavailable.
- CPU timing declares its boundary, uses warm-up, measures repeated per-frame perception at 640x360, and reports median and p95 honestly whether or not the 50 ms target passes.

---

### Task 1: Frozen configuration and output-contract projection

**Files:**
- Create: `66_crop_row_offline_video_pilot/code/day66_frozen_config.json`
- Create: `66_crop_row_offline_video_pilot/tests/test_day66_offline_video_pilot.py`
- Create: `66_crop_row_offline_video_pilot/code/day66_offline_video_pilot.py`

**Interfaces:**
- Consumes: accepted `day65_results.json`, Day65 `TemporalConfig`, Day65 frame records.
- Produces: `load_and_verify_frozen_config(config_path, day65_result_path) -> tuple[TemporalConfig, bool, dict]` and `project_navigation_contract(record) -> dict`.

- [x] **Step 1: Write failing tests** for exact frozen values, mismatch rejection, candidate-to-degraded mapping, and null navigation fields outside `valid`.
- [x] **Step 2: Run focused pytest with a new project-local `--basetemp`** and verify failure is caused by the missing Day66 module.
- [x] **Step 3: Add the selected Day65 JSON values** and the minimal loader/projection implementation. Verify both file SHA-256 values are recorded and `TemporalConfig()` defaults are never used implicitly.
- [x] **Step 4: Re-run focused tests** and require all Task 1 tests to pass.

### Task 2: Continuous, truthful overlay rendering

**Files:**
- Modify: `66_crop_row_offline_video_pilot/tests/test_day66_offline_video_pilot.py`
- Modify: `66_crop_row_offline_video_pilot/code/day66_offline_video_pilot.py`

**Interfaces:**
- Consumes: source BGR frame and projected Day66 record.
- Produces: `draw_pilot_overlay(frame, record) -> np.ndarray` with row IDs, selected left/right boundaries, valid-only center, confidence, pilot/public state, and reason.

- [x] **Step 1: Add failing image tests** proving row-ID labels and selected boundaries alter expected regions while candidate/degraded/reject frames never draw a navigation center.
- [x] **Step 2: Run the new tests and observe expected failures.**
- [x] **Step 3: Implement the overlay** with fixed color semantics and bounded text layout suitable for 320x180 source videos.
- [x] **Step 4: Re-run focused tests** and require green.

### Task 3: Episode and manifest pilot runner

**Files:**
- Modify: `66_crop_row_offline_video_pilot/tests/test_day66_offline_video_pilot.py`
- Modify: `66_crop_row_offline_video_pilot/code/day66_offline_video_pilot.py`

**Interfaces:**
- Consumes: one manifest development entry, frozen predictor/config, output directory.
- Produces: `run_episode_pilot(...) -> tuple[list[dict], dict]`, playable overlay MP4, frame-aligned JSONL, and `run_day66_pilot(...) -> dict`.

- [x] **Step 1: Add failing tiny-video integration tests** for exact frame alignment, readable output video, fresh tracker state, development-role selection, frozen-role non-access, checksums, and incomplete-decode failure.
- [x] **Step 2: Run the integration tests and observe expected failures.**
- [x] **Step 3: Implement the minimum second-pass overlay/JSONL writer and aggregate runner.** Preserve Day65 perception behavior and add no tuning knobs for temporal parameters.
- [x] **Step 4: Add acceptance checks** for 25 episodes, 10,995 frames, all complete decodes, all output videos readable, navigation invariants, exact config match, allowed roles only, and frozen access false.
- [x] **Step 5: Re-run Day66 tests** and require green.

### Task 4: Reproducible CPU runtime benchmark and CLI

**Files:**
- Modify: `66_crop_row_offline_video_pilot/tests/test_day66_offline_video_pilot.py`
- Modify: `66_crop_row_offline_video_pilot/code/day66_offline_video_pilot.py`

**Interfaces:**
- Produces: `benchmark_canonical_cpu(...) -> dict`, CLI arguments limited to paths/device/batch/render/benchmark controls, and marker output.

- [x] **Step 1: Add failing tests** for warm-up exclusion, finite median/p95, declared 640x360 boundary, and absence of mutable Day65 threshold CLI flags.
- [x] **Step 2: Run and observe expected failures.**
- [x] **Step 3: Implement the benchmark and CLI.** Benchmark inference plus temporal measurement only, excluding video decode and overlay encoding; use batch size one and `torch.inference_mode()` through the frozen predictor.
- [x] **Step 4: Re-run Day66 and Day65 regression tests.**

### Task 5: Smoke run and complete 25-video development pilot

**Files:**
- Create locally only: `D:/DL_code/data/crop_row_perception/day66_offline_video_pilot/**`

**Interfaces:**
- Consumes: CROW manifest and frozen Day63 checkpoint.
- Produces: 25 overlay MP4s, 25 JSONLs, `day66_results.json`, runtime evidence, and audit candidates.

- [x] **Step 1: Run one short development episode smoke test** with CUDA and verify video/frame/JSON alignment without changing parameters.
- [x] **Step 2: Inspect representative rendered frames** for legibility, row IDs, boundary selection, valid-only center, and status meaning. Fix only rendering/contract defects under new failing tests.
- [x] **Step 3: Run the complete 25-video pilot** using the frozen configuration.
- [x] **Step 4: Run the canonical CPU benchmark** on a preregistered development clip and retain the result even if the 50 ms threshold fails.
- [x] **Step 5: Validate output counts, hashes, decodability, aggregate checks, and frozen non-access.** If a check fails, preserve the failed attempt, add a regression test, and make only a development-bounded correction.

### Task 6: Visual audit and Day66 learning note

**Files:**
- Create: `66_crop_row_offline_video_pilot/code/day66_notes.md`
- Create: `66_crop_row_offline_video_pilot/assets/README.md`

**Interfaces:**
- Consumes: verified Day66 results and stratified samples.
- Produces: reviewable contact sheets and a Chinese evidence-bounded lesson record.

- [x] **Step 1: Generate deterministic samples** stratified by episode and `valid/candidate/degraded/reject`, plus switch/flow/rejection transitions.
- [x] **Step 2: Inspect the actual contact sheets and at least two complete overlay videos.** Record visible successes and failures without using unlabeled visual impressions as accuracy.
- [x] **Step 3: Write `day66_notes.md`** with objective, architecture, frozen configuration, TDD, complete-run metrics, runtime, visual audit, failed attempts, evidence boundaries, reproduction command, local artifacts, and Day67 handoff.
- [x] **Step 4: Write the asset README** identifying local-only outputs and licensing restrictions.

### Task 7: Final verification and review-gated delivery

**Files:**
- Verify all new Day66 tracked files; modify root/project README and deliver only after user review authorizes it.

- [x] **Step 1: Run all Day63–66 tests** with a fresh project-local `--basetemp`.
- [x] **Step 2: Run syntax compilation and the Day66 artifact verifier.**
- [x] **Step 3: Re-run the learning repository scanner** and verify latest Day is 66 with no missing days.
- [x] **Step 4: Inspect `git diff --check`, `git status`, and the full Day66 diff.** Confirm no data, cache, model, or generated bulk artifact is tracked.
- [x] **Step 5: Stop for user review, then commit and push only after explicit authorization.** The user approved delivery on 2026-09-07.

## Self-Review

- Spec coverage: full pilot, overlays, row identities, corridor center, confidence, four diagnostic states, exact frozen config, development-only scope, runtime, visual QA, notes, and review-gated GitHub upload are each assigned to a task.
- Evidence boundary: frozen same-source videos remain structurally excluded; real-video safety, metric calibration, external generalization, and robot readiness remain blocked.
- Placeholder scan: no implementation requirement is deferred; Day67 failure grouping is intentionally outside Day66.
- Type consistency: Task 1 produces projected records consumed by Tasks 2–6; Task 3 produces aggregate evidence consumed by Tasks 5–7.
