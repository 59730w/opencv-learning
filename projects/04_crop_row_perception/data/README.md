# 数据目录说明

仓库只保存来源登记与审查结果，不保存第三方原始图像。

Day60 下载并实审的 49.7 MB 试审包原位于：

```text
D:\DL_code\data\crop_row_perception\sources\zenodo_18700580\extracted\S1_File\
```

原始 `S1_File.zip` 在完成 MD5 校验和解压审查后已按用户要求删除，当前只保留上述 80 对解压样本。

官方下载页：<https://zenodo.org/records/18700580>，许可为 CC BY 4.0，归档 MD5 为 `79d2b2327e2389da496be8a40d6db415`。

该包只允许用于格式、标签转换和最小管线试运行。它混合 CRBD 子集与作者采集图像，却没有逐文件来源/序列清单，也不含应拒绝负样本，因此不能随机划分后充当 ID、OOD 或冻结外部证据。

## 下载后状态

- CRDLD 已下载并解压到 `D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\`；1,930 对通过配对与解码检查。由于许可、最高分组和负样本仍缺失，只能作受限 ID 开发候选。
- RowDetr 已下载并解压到 `D:\DL_code\data\crop_row_perception\sources\rowdetr_kaggle_8432059\staging\crop-row-detection\`；其中 1,760 张干净的高粱机器人图像已冻结为外部正样本，清单见 `frozen_external_sorghum_manifest.json`。
- Sugar Beets 2016 只做了远程 ZIP 抽样。Kinect 和 JAI RGB 都是近垂直俯视，已否决为前视拒识负样本；完整包没有保留。
- CCRDNet 暂不下载，Bonn Multi-Crop 暂不发送申请邮件。

CRDLD、RowDetr 的原始 ZIP、CRDLD 内部重复 ZIP，以及 Sugar Beets 中断碎片均在哈希/解压/抽样验证后删除。仓库仍只保存来源登记、审查结果与冻结清单，不保存第三方原始图像。

正式状态、哈希与阻断项见 `source_registry.json` 和 `downloaded_sources_audit.json`。完整数据门禁仍为 `BLOCKED`：当前缺少前视目标域应拒绝负样本，且 CRDLD 的许可和最高泄漏分组未解决。
