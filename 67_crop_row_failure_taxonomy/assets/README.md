# Day67 assets

本目录不提交原始视频、Day66 JSONL、复核联系表、上下文视频或逐事件标注。LeCropFollow CROW
的数据许可尚未核验，Day67生成证据仅保存在本机：

```text
D:\DL_code\data\crop_row_perception\day67_failure_taxonomy\
```

主要本地产物：

- `day67_events.jsonl`：10,995帧折叠出的4,809个连续系统事件；
- `day67_review_manifest_v3.jsonl`：固定种子6703生成的536事件复核清单，包含全部296个valid事件和240个层内等概率无放回样本；
- `day67_annotations_v3.jsonl`：两轮逐事件复核与重点上下文复核后的开发期标注；
- `day67_results_v3.json`：完整事件统计、设计加权视觉类别统计、疑似不安全valid和Day68决策；
- `review_sheets_v3/`、`review_nonvalid_v3/`：34张全量与15张non-valid原图/Day66叠加联系表；
- `review_clips_v3/`：41个疑似OOD valid和严重遮挡事件的逐帧上下文证据；
- 第一、二版文件：保留未加权与权重错配的失败记录，不用于Day68决策。

视觉标注是 `MODEL_ASSISTED_REVIEW_DEVELOPMENT_ONLY`，不是独立人工真值，也不能用于计算真实视频
准确率或安全率。六段冻结同源视频未被打开、解码、推理、渲染或标注。
