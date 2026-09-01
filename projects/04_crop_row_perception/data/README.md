# 数据目录说明

仓库只保存来源登记与审查结果，不保存第三方原始图像。

Day60 下载并实审的 49.7 MB 试审包原位于：

```text
D:\DL_code\data\crop_row_perception\sources\zenodo_18700580\extracted\S1_File\
```

原始 `S1_File.zip` 在完成 MD5 校验和解压审查后已按用户要求删除，当前只保留上述 80 对解压样本。

官方下载页：<https://zenodo.org/records/18700580>，许可为 CC BY 4.0，归档 MD5 为 `79d2b2327e2389da496be8a40d6db415`。

该包只允许用于格式、标签转换和最小管线试运行。它混合 CRBD 子集与作者采集图像，却没有逐文件来源/序列清单，也不含应拒绝负样本，因此不能随机划分后充当 ID、OOD 或冻结外部证据。

正式三角色数据的下载状态、链接与阻断项见 `source_registry.json`。在数据门禁通过前，不下载 1.4 GB CRDLD 或 6.4 GB CCRDNet 大包，不开始模型开发。
