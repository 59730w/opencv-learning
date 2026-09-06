# Day65 assets

Day65 的原始视频、逐帧 JSONL、结果 JSON、抽样图和叠加视频保存在本地实验目录，
不纳入 Git。

原因有两点：其一，LeCropFollow CROW 数据卡没有给出已核验的数据再分发许可；其二，
逐帧结果和视频体积较大。仓库只保留可复现代码、测试和文字笔记。

本地结果目录：

```text
D:\DL_code\data\crop_row_perception\day65_video_temporal_verified
D:\DL_code\data\crop_row_perception\day65_video_temporal_final
```

其中 `verified` 是从原视频和冻结 Day63 模型通过正式 CLI 完整复跑所得；`final` 还包含
24 帧有效状态抽样图和两段可视化叠加视频。
