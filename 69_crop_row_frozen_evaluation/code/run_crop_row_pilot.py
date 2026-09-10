"""User-facing offline crop-row pilot for one video or a directory of videos."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for code_dir in (
    Path(__file__).resolve().parent,
    PROJECT_ROOT / "66_crop_row_offline_video_pilot" / "code",
    PROJECT_ROOT / "68_crop_row_severe_occlusion" / "code",
):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from day66_offline_video_pilot import build_execution_predictor, load_and_verify_frozen_config  # noqa: E402
from day68_severe_occlusion import OcclusionGuardConfig  # noqa: E402
from day69_frozen_evaluation import run_single_video  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
DEFAULT_CHECKPOINT = Path(r"D:\DL_code\data\crop_row_perception\day63_crop_row_geometry\day63_resnet18_centerline_model.pt")
DEFAULT_DAY65_RESULT = Path(r"D:\DL_code\data\crop_row_perception\day65_video_temporal_verified\day65_results.json")
DEFAULT_TEMPORAL_CONFIG = PROJECT_ROOT / "66_crop_row_offline_video_pilot" / "code" / "day66_frozen_config.json"
DEFAULT_OCCLUSION_CONFIG = PROJECT_ROOT / "68_crop_row_severe_occlusion" / "code" / "day68_frozen_config.json"


def discover_video_inputs(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported video extension: {path.suffix}")
        return [path.resolve()]
    if path.is_dir():
        videos = sorted(
            item.resolve() for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise ValueError(f"no supported videos found under: {path}")
        return videos
    raise ValueError(f"input does not exist: {path}")


def _load_occlusion_config(path: Path) -> OcclusionGuardConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("marker") != "DAY68_FROZEN_CONFIG":
        raise ValueError("occlusion config is not the accepted Day68 freeze")
    names = {item.name for item in fields(OcclusionGuardConfig)}
    return OcclusionGuardConfig(**{name: payload[name] for name in names})


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    videos = discover_video_inputs(args.input)
    temporal, optical_flow, temporal_evidence = load_and_verify_frozen_config(
        args.temporal_config, args.day65_result
    )
    occlusion = _load_occlusion_config(args.occlusion_config)
    predictor = build_execution_predictor(
        args.checkpoint, device=args.device, batch_size=args.batch_size
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    used_names: dict[str, int] = {}
    for video in videos:
        base = video.stem
        suffix = used_names.get(base, 0)
        used_names[base] = suffix + 1
        episode = base if suffix == 0 else f"{base}_{suffix}"
        reports.append(run_single_video(
            video, output_dir=output_dir / episode, predictor=predictor,
            temporal_config=temporal, occlusion_config=occlusion,
            use_optical_flow=optical_flow, episode=episode,
            evidence_role="user_input_unverified",
        ))
    aggregate = {
        "marker": "DAY69_OFFLINE_PILOT_COMPLETE",
        "input": str(Path(args.input).resolve()),
        "video_count": len(reports),
        "frame_count": sum(item["source_frame_count"] for item in reports),
        "navigation_invariant_violations": sum(item["navigation_invariant_violations"] for item in reports),
        "temporal_freeze_evidence": temporal_evidence,
        "device": predictor.device,
        "predictor_backend": getattr(predictor, "backend", "torch_eager"),
        "reports": reports,
        "claim_boundary": "Offline diagnostic pilot only; outputs are not field-safe navigation commands.",
    }
    report_path = output_dir / "pilot_report.json"
    report_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate["report_path"] = str(report_path)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="video file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--temporal-config", type=Path, default=DEFAULT_TEMPORAL_CONFIG)
    parser.add_argument("--day65-result", type=Path, default=DEFAULT_DAY65_RESULT)
    parser.add_argument("--occlusion-config", type=Path, default=DEFAULT_OCCLUSION_CONFIG)
    return parser


def main() -> int:
    result = run_pilot(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
