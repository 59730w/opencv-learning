"""Local click-to-run web interface for the frozen Day69 offline pilot."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from run_crop_row_pilot import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DAY65_RESULT,
    DEFAULT_OCCLUSION_CONFIG,
    DEFAULT_TEMPORAL_CONFIG,
    VIDEO_EXTENSIONS,
    run_pilot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / ".tmp" / "day69-web-runs"


@dataclass(frozen=True)
class WebRunResult:
    overlay_video: Path
    status_markdown: str
    summary: dict[str, Any]
    downloads: tuple[Path, ...]
    output_dir: Path


def resolve_device(label: str) -> str | None:
    mapping = {"自动选择": None, "CPU": "cpu", "CUDA": "cuda"}
    if label not in mapping:
        raise ValueError(f"未知设备选项：{label}")
    return mapping[label]


def _safe_video_stem(path: Path) -> str:
    ascii_stem = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-").lower()
    if any(ord(char) > 127 for char in path.stem):
        ascii_stem = f"field-{ascii_stem}" if ascii_stem else "field-video"
    return ascii_stem or "field-video"


def _notify(progress: Callable[..., Any] | None, value: float, description: str) -> None:
    if progress is not None:
        progress(value, desc=description)


def run_uploaded_video(
    uploaded_video: str | Path | None,
    device_label: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    runner: Callable[[argparse.Namespace], dict[str, Any]] = run_pilot,
    run_id: str | None = None,
    progress: Callable[..., Any] | None = None,
) -> WebRunResult:
    """Validate one uploaded video, run the canonical pilot, and expose its artifacts."""
    if uploaded_video is None:
        raise ValueError("请先选择视频，再点击开始处理。")
    video_path = Path(uploaded_video).resolve()
    if not video_path.is_file():
        raise ValueError(f"找不到上传的视频：{video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"不支持的视频格式。请选择：{supported}")

    identifier = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = Path(output_root).resolve() / f"{identifier}-{_safe_video_stem(video_path)}"
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖已停止：{output_dir}")

    _notify(progress, 0.05, "正在校验冻结配置")
    args = argparse.Namespace(
        input=video_path,
        output_dir=output_dir,
        device=resolve_device(device_label),
        batch_size=None,
        checkpoint=DEFAULT_CHECKPOINT,
        temporal_config=DEFAULT_TEMPORAL_CONFIG,
        day65_result=DEFAULT_DAY65_RESULT,
        occlusion_config=DEFAULT_OCCLUSION_CONFIG,
    )
    _notify(progress, 0.12, "正在检测、跟踪并生成叠加视频")
    aggregate = runner(args)
    reports = aggregate.get("reports") or []
    if len(reports) != 1:
        raise RuntimeError("网页单视频任务必须返回且仅返回一份视频报告。")

    report = reports[0]
    required = {
        "overlay_video": Path(report["overlay_video"]),
        "output_csv": Path(report["output_csv"]),
        "output_jsonl": Path(report["output_jsonl"]),
        "report_path": Path(report["report_path"]),
        "pilot_report": Path(aggregate["report_path"]),
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"处理完成但缺少输出文件：{', '.join(missing)}")
    if not report.get("overlay_verification", {}).get("passed"):
        raise RuntimeError("叠加视频未通过完整解码验证。")

    counts = report.get("status_counts", {})
    frame_count = int(aggregate.get("frame_count", 0))
    violations = int(aggregate.get("navigation_invariant_violations", 0))
    status_markdown = (
        "### 处理完成\n\n"
        f"**{frame_count} 帧** · valid **{int(counts.get('valid', 0))}** · "
        f"candidate **{int(counts.get('candidate', 0))}** · "
        f"degraded **{int(counts.get('degraded', 0))}** · "
        f"reject **{int(counts.get('reject', 0))}**\n\n"
        f"导航规则违规：**{violations}**。仅 `valid` 状态可以输出走廊中心；"
        "其他状态只用于诊断。"
    )
    summary = {
        "status": "complete",
        "frame_count": frame_count,
        "status_counts": counts,
        "navigation_invariant_violations": violations,
        "overlay_decode_complete": True,
        "device": aggregate.get("device"),
        "predictor_backend": aggregate.get("predictor_backend"),
        "output_dir": str(output_dir),
        "claim_boundary": aggregate.get("claim_boundary"),
    }
    _notify(progress, 1.0, "完成，可以查看结果")
    return WebRunResult(
        overlay_video=required["overlay_video"],
        status_markdown=status_markdown,
        summary=summary,
        downloads=tuple(required.values()),
        output_dir=output_dir,
    )


def process_for_ui(
    uploaded_video: str | None,
    device_label: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str, dict[str, Any], list[str], str]:
    try:
        result = run_uploaded_video(uploaded_video, device_label, progress=progress)
    except Exception as error:
        raise gr.Error(str(error)) from error
    return (
        str(result.overlay_video),
        result.status_markdown,
        result.summary,
        [str(path) for path in result.downloads],
        f"结果保存在：`{result.output_dir}`",
    )


CSS = """
:root {
  --field-deep: #12372f;
  --field-night: #09261f;
  --field-leaf: #39785a;
  --soil: #a56f49;
  --signal: #f2c85b;
  --paper: #f4f2e8;
  --ink: #17251f;
  --mist: #dce5dc;
}
.gradio-container {
  max-width: 1180px !important;
  background:
    radial-gradient(circle at 86% 2%, rgba(57,120,90,.15), transparent 25rem),
    var(--paper) !important;
  color: var(--ink) !important;
  font-family: "Microsoft YaHei UI", "Noto Sans SC", sans-serif !important;
}
.heji-hero {
  position: relative;
  overflow: hidden;
  min-height: 260px;
  padding: 38px 42px !important;
  border: 0 !important;
  border-radius: 30px !important;
  color: #f8f4e8;
  background:
    linear-gradient(112deg, rgba(9,38,31,.98) 0 56%, rgba(18,55,47,.78)),
    repeating-linear-gradient(74deg, transparent 0 46px, rgba(242,200,91,.62) 48px 51px, transparent 53px 92px);
  box-shadow: 0 24px 70px rgba(9,38,31,.22);
}
.heji-hero::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(0deg, transparent 0 27px, rgba(255,255,255,.035) 28px 29px);
}
.heji-hero h1 { font-size: clamp(50px, 8vw, 88px); line-height: .9; margin: 12px 0 18px; letter-spacing: -.08em; font-weight: 800; }
.heji-hero p { max-width: 650px; font-size: 16px; line-height: 1.8; color: rgba(248,244,232,.80); }
.heji-kicker { font-family: "Cascadia Code", monospace; color: var(--signal); letter-spacing: .2em; font-size: 11px; }
.heji-seal { position: absolute; right: 34px; top: 30px; border: 1px solid rgba(242,200,91,.65); border-radius: 999px; padding: 8px 13px; font: 11px "Cascadia Code", monospace; letter-spacing: .12em; color: var(--signal); }
.heji-panel { background: rgba(255,255,255,.76) !important; border: 1px solid rgba(18,55,47,.12) !important; border-radius: 24px !important; padding: 20px !important; box-shadow: 0 12px 40px rgba(18,55,47,.07); backdrop-filter: blur(12px); }
.heji-warning { border-left: 5px solid var(--signal) !important; background: #fff8dc !important; border-radius: 12px !important; padding: 12px 16px !important; }
.heji-warning p { color: #493a16 !important; }
.heji-warning code { color: #fff8dc !important; background: var(--field-night) !important; }
.heji-panel h2, .heji-panel > div > p { color: var(--ink) !important; }
.heji-step { font-family: "Cascadia Code", monospace; color: var(--field-leaf); font-size: 11px; letter-spacing: .12em; }
.primary { background: var(--field-deep) !important; border-color: var(--field-deep) !important; }
.primary:hover { background: var(--field-leaf) !important; }
.heji-result video { border-radius: 18px !important; background: var(--field-night) !important; }
button, input, textarea { transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease !important; }
button.primary:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(18,55,47,.18) !important; }
footer { display: none !important; }
@media (max-width: 720px) { .heji-hero { padding: 30px 22px !important; border-radius: 20px !important; } .heji-seal { display: none; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
    )
    with gr.Blocks(theme=theme, css=CSS, title="禾迹 · 作物行视觉台") as app:
        gr.HTML(
            """
            <section class="heji-hero">
              <div class="heji-kicker">HEJI · LOCAL FIELD VISION</div>
              <div class="heji-seal">LOCAL / PRIVATE</div>
              <h1>禾迹</h1>
              <p>让每一条作物行留下可检查的轨迹。上传田间视频，本地视觉流程将完成检测、跟踪、走廊判断与逐帧记录。</p>
            </section>
            """
        )
        gr.Markdown(
            "**安全边界：** 只有 `valid` 才显示可用走廊中心。"
            "`candidate / degraded / reject` 保留诊断信息，但绝不作为导航输出。",
            elem_classes="heji-warning",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="heji-panel"):
                gr.HTML('<div class="heji-step">INPUT · 输入</div>')
                gr.Markdown("## 选择现场视频")
                video_input = gr.Video(
                    label="点击上传或拖入视频",
                    sources=["upload"],
                    format="mp4",
                    height=330,
                )
                device = gr.Radio(
                    choices=["自动选择", "CUDA", "CPU"],
                    value="自动选择",
                    label="运行设备",
                    info="自动选择会优先使用可用的 NVIDIA GPU。",
                )
                run_button = gr.Button("开始处理视频", variant="primary", size="lg")
                gr.Markdown("视频只在本机处理；页面不会创建公网分享链接。")
            with gr.Column(scale=6, elem_classes="heji-panel heji-result"):
                gr.HTML('<div class="heji-step">OUTPUT · 结果</div>')
                gr.Markdown("## 查看处理结果")
                result_video = gr.Video(label="检测与跟踪叠加视频", interactive=False, height=330)
                status = gr.Markdown("上传视频后，处理统计会显示在这里。")

        with gr.Accordion("报告与下载", open=True):
            summary = gr.JSON(label="本次运行摘要")
            downloads = gr.File(label="下载全部结果", file_count="multiple", interactive=False)
            output_location = gr.Markdown()

        run_button.click(
            fn=process_for_ui,
            inputs=[video_input, device],
            outputs=[result_video, status, summary, downloads, output_location],
            api_name=False,
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7869)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=not args.no_browser,
        show_error=True,
        allowed_paths=[str(DEFAULT_OUTPUT_ROOT.resolve())],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
