# Day66 assets

本目录不提交原始视频、逐帧JSONL、叠加MP4或联系表。Day66使用的LeCropFollow CROW数据许可
尚未核验，所有生成媒体只保存在：

```text
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\
```

主要本地产物：

- `day66_results.json`：完整Pilot、源码/模型/输入哈希、结构门和CPU基准；
- `day66_visual_audit.json`：确定性分层抽样索引；
- `*_day66_overlay.mp4`：25段连续叠加视频；
- `*.jsonl`：25段逐帧合同输出；
- `day66_audit_*.jpg`：四状态、状态切换和光流补测联系表；
- `attempt_01_runtime_failed.json`：首轮CPU门失败记录；
- `cpu_thread_diagnostic.json`、`cpu_backend_optimization.json`：不改变权重/阈值的运行时间诊断。

这些媒体只用于本地学习与审计，不作为可重新分发的数据资产。真实视频缺少逐帧走廊有效性真值，
视觉抽样只能发现明显问题，不能计算准确率或安全率。
