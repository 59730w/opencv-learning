# Day60 数据可行性审查报告

审查日期：2026-09-01

复核日期：2026-09-01

结论：`BLOCKED`
门禁标识：`DAY60_DATA_VIABILITY_BLOCKED`

## 1. 结论先行

CRDLD 与 RowDetr 已完成下载、解压、哈希、配对、解码、标签和跨来源重复审查。现在已经具备：

- 1,930 对 CRDLD 前视机器人图像/掩码，可作受限的 ID 开发候选；
- 1,760 张通过标签质检的 RowDetr 高粱机器人图像，可冻结为独立来源的**外部正样本几何测试**；
- 可复现的来源审查器与冻结清单。

但整个数据门禁仍不能判为 `PASS`，因为 CRDLD 没有明确的数据许可证和田块/日期/序列/相机分组，两个正式数据源都不含应拒绝负样本。外部正样本可用，不等于外部拒识和完整泛化已经可验证。

## 2. 下载后候选结论

| 候选 | 实际角色 | 下载后证据 | 仍有限制 | 结论 |
|---|---|---|---|---|
| CRDLD v2.1 | ID 开发候选 | 1,930 对全部可解码且配对；前视 Husky 场景；发现 1 组跨官方划分完全重复 | 随包 README 为空；无明确数据许可；无最高泄漏分组；无拒绝负样本 | `AVAILABLE_WITH_RESTRICTIONS`，不能据官方随机/类别划分声明泛化 |
| RowDetr 高粱子源 | 冻结外部正样本 | 来源与 CRDLD 独立；1,783 张机器人图像；无 CRDLD 精确重复 | 23 张标签不合法被排除；序列键不可恢复；无负样本 | 冻结 1,760 张，只评估正样本位置/方向 |
| Sugar Beets 2016 | 否决的负样本候选 | 官方 CC BY-SA 4.0；按序列组织；远程抽查 Kinect 24 帧、JAI RGB 12 帧 | 两个相机均为垂直俯视，不能代表前视导航拒识场景 | `REJECT_VIEWPOINT_MISMATCH` |
| CCRDNet | 暂缓的 OOD 候选 | CC BY 4.0、多作物中央行掩码 | 手持视角、无分组清单、无负样本 | 暂不下载 |
| Bonn Multi-Crop | 未申请候选 | 任务和指标匹配 | 需邮件申请、许可与文件级分组未确认 | 不作为当前可执行方案 |

机器可读记录见 `data/source_registry.json`，下载实审结果见 `data/downloaded_sources_audit.json`。

## 3. CRDLD 实审结果

原始下载地址：<https://lcas.lincoln.ac.uk/nextcloud/index.php/s/Eip4nWbetxJQ6No/download>

- 原始归档大小：1,463,296,501 字节；
- SHA-256：`402fb6a2a236996fbdbd685c53b8945c5fa4f51798117127f63135153bcae4e7`；
- 实际划分：train 1,250、test 430、validation 250，共 1,930 对，并非页面描述的 2,000 对；
- 所有图像/标签均配对、可解码且尺寸相同；
- 标签为有 JPEG 压缩损失的灰度掩码，不能按严格 `0/255` 二值读取；
- `train_data/687` 与 `test_data/276` 是完全相同图像；
- 随包 `Readme.md` 只有 1 字节，未提供数据许可证或田块/日期/序列/相机键。

本地有效数据：`D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\`。归档与重复内部 ZIP 已在哈希和解压验证后删除。

## 4. RowDetr 外部正样本实审

来源页：<https://www.kaggle.com/datasets/rahulharsha/crop-row-detection>，许可为 CC BY-NC-SA 4.0。

- 原始归档大小：3,905,613,297 字节；
- MD5：`16ef0ecba4e774be790a8530740bfa3a`，与远端 ETag 一致；
- SHA-256：`50bb3b9ba608492e1bf7a1ff544c73e72eafdffd39cc9e9dbcfe71c6d10718ea`；
- 通用 train/val/test 共 6,369 对，不是论文报告的 6,962 对；
- 其中 720x1280 的高粱机器人子源共 1,783 张；
- 23 张存在折线点数不足或坐标越界，排除后冻结 1,760 张；
- 与本地 CRDLD 没有精确图像重复；
- 不含空标签或真实拒绝负样本；文件名为划分内数字编号，不能恢复原始序列键。

冻结清单：`data/frozen_external_sorghum_manifest.json`。整个清单只允许在模型、预处理、阈值和置信度规则冻结后使用，不能反向调参，也不能用于证明拒识能力。

本地有效数据：`D:\DL_code\data\crop_row_perception\sources\rowdetr_kaggle_8432059\staging\crop-row-detection\`。原始 ZIP 已在校验与解压后删除。

## 5. Sugar Beets 2016 为什么不能补负样本

官方来源：<https://www.ipb.uni-bonn.de/data/sugarbeets2016/index.html>，许可为 CC BY-SA 4.0。候选序列 `bonirob_2016-04-25-17-16-29_0` 的完整归档约 1.7 GB。

为避免无意义下载，使用 HTTP Range 只读取远程 ZIP 目录并均匀抽查：Kinect color 24/248 帧、JAI RGB 12/252 帧。两组画面都近乎垂直朝向地面，Kinect 可见轮子/底盘，JAI RGB 主要是暗色土壤。这类负样本与目标前视相机域差距太大，模型仅凭视角就能拒绝，不能构成有挑战性的目标域负样本。

抽样证据保留在 `D:\DL_code\data\crop_row_perception\sources\sugarbeets2016_negatives\`，完整归档未下载；中断碎片已删除。

## 6. 当前最合理的后续方案

1. CRDLD 只作为受限 ID 开发数据；首次使用时删除跨划分重复项，并明确“不声称无泄漏内部泛化”。
2. RowDetr 高粱 1,760 张作为一次性冻结外部正样本，不参与任何开发决策。
3. 单独建立前视目标域拒识集，优先收集或取得有明确许可的视频序列，覆盖地头、无行、非农田、严重模糊、过曝/欠曝和完全遮挡；按完整视频/日期分组，不能从正样本裁剪伪造。
4. 在真实拒识集到位前，可以学习数据加载、掩码清理和几何标签转换，但不得报告完整外部泛化通过，也不得把数据门禁改成 `PASS`。
5. CCRDNet 暂不下载；Bonn 邮件申请仍可作为以后扩大证据的备选，而不是当前进度的前置条件。

## 7. 可复现命令

```powershell
D:\conda\envs\forest-species\python.exe projects\04_crop_row_perception\data\audit_downloaded_sources.py `
  --crdld-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data `
  --rowdetr-root D:\DL_code\data\crop_row_perception\sources\rowdetr_kaggle_8432059\staging\crop-row-detection `
  --output projects\04_crop_row_perception\data\downloaded_sources_audit.json

D:\conda\envs\forest-species\python.exe projects\04_crop_row_perception\data\build_frozen_sorghum_manifest.py `
  --rowdetr-root D:\DL_code\data\crop_row_perception\sources\rowdetr_kaggle_8432059\staging\crop-row-detection `
  --output projects\04_crop_row_perception\data\frozen_external_sorghum_manifest.json
```

当前机器审查结论仍为 `BLOCKED`，原因是许可/最高泄漏分组和目标域拒绝负样本尚未同时满足。
