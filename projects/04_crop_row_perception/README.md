# Practical 04：农业机器人作物行视觉感知

这是一个 Day59-Day70 的短周期学习项目，目标是把已有 OpenCV、深度学习、C++和实验验证能力迁移到农业机器人前视机器视觉问题。

当前只研究离线视觉感知：从单目 RGB 图像或视频帧估计中央作物行/可通行走廊的图像平面中心偏移、局部方向、置信度和拒绝状态。项目不包含真实机器人控制，也不把图像坐标误写成物理距离。

CRDLD 等公开数据目前仅是待审查候选，尚未下载或实测。内部与冻结外部必须同时通过绝对阈值和差距规则，才能形成泛化成功声明。

## 门禁状态

| 门禁 | 状态 | 日期 | 证据 |
|---|---|---|---|
| 目标契约 | PASS | 2026-08-31 | `target_contract.yaml` |
| 数据可行性 | BLOCKED | 2026-09-01 | `data_viability_report.md`：许可、分组与冻结外部证据不足 |
| 环境 | NOT_AVAILABLE | — | 前两门通过后执行 |
| 管线试运行 | NOT_AVAILABLE | — | 后续执行 |
| 内部有效性 | NOT_AVAILABLE | — | 后续执行 |
| 基线/OOD开发 | NOT_AVAILABLE | — | 后续执行 |
| 冻结外部测试 | NOT_AVAILABLE | — | 后续执行 |
| 交付 | NOT_AVAILABLE | — | Day70 执行 |

只有显式 `PASS` 才能进入下一门。开源项目展示和论文指标均不算本项目效果证据。

Day60 的审查工作已经完成，但数据门禁为 `BLOCKED`。这表示当天学习已完成，不表示项目可以继续进入环境或管线开发。

## Day59 产物

- `target_contract.yaml`：任务、坐标、输出、条件、指标和声明边界；
- `evidence_registry.yaml`：开发、OOD与冻结外部证据角色；
- `docs/open_source_baseline_review.md`：七个开源候选的适用性与许可证边界；
- `../../59_crop_row_perception_contract/code/day59_geometry.py`：坐标与误差定义的最小可执行参考；
- `../../59_crop_row_perception_contract/code/day59_notes.md`：完整中文学习笔记。

## Day60 产物

- `data_viability_report.md`：候选数据、实下载检查、阻断项与解除条件；
- `data/source_registry.json`：五个候选的机器可读来源登记；
- `data/audit_result.json`：49.7 MB 试审包的配对、解码、尺寸、标签值和重复检查；
- `../../60_crop_row_data_viability/code/day60_data_audit.py`：可重跑的数据审查器；
- `../../60_crop_row_data_viability/code/day60_notes.md`：完整中文学习笔记。
