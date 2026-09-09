# Day68 assets

本目录不提交原始视频、逐帧JSONL、叠加视频或审查截图。LeCropFollow CROW的数据许可尚未
核验，Day68生成证据仅保存在本机：

```text
D:\DL_code\data\crop_row_perception\day68_severe_occlusion\
```

本地产物包括：

- `day68_results.json`：正式指标、逐段输出哈希、验收门和证据边界；
- `*_day68.jsonl`：25段开发视频的帧对齐结果；
- `*_day68_overlay.mp4`：25段连续诊断叠加视频；
- 共10,995帧，所有叠加视频都已再次完整解码并核对帧数。

六段冻结同源测试视频未被打开、解码、推理、渲染或标注。临时探索和视觉审查图位于
`.tmp/`，不会随Day68上传。
