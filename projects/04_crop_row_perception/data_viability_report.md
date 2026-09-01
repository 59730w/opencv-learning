# Day60 数据可行性审查报告

审查日期：2026-09-01
结论：`BLOCKED`
门禁标识：`DAY60_DATA_VIABILITY_BLOCKED`

## 1. 结论先行

Day60 的学习与审查步骤已经完成，但项目的数据可行性门禁没有通过。当前没有一套证据组合能同时满足：地面机器人前视视角、许可清楚、标签支持位置与方向指标、最高泄漏单位可分组、含应拒绝负样本，以及独立冻结外部来源。

因此今天不进入环境搭建或算法管线，不把任何官方 Train/Test 目录、连续帧随机切分或不同分辨率子集包装成外部泛化。

## 2. 候选结论

| 候选 | 预期角色 | 优点 | 阻断项 | 结论 |
|---|---|---|---|---|
| CRDLD v2.1 | ID 开发 | Husky 前视、2000 对图像/掩码、50 条件类、线坐标 | 数据许可未写明；田块/日期/序列/相机组未知；无拒绝负样本 | 最匹配但暂缓下载 |
| CCRDNet | OOD 开发 | CC BY 4.0、中央行三类掩码、8 作物、直链 | 手持手机且高度/角度不受控；无负样本；归档无分组清单 | 只作 OOD 候选 |
| Bonn Multi-Crop | 冻结外部 | 多作物、机器人视角、位置/方向评价完全匹配 | 需邮件申请；数据许可未单独确认；未拿到文件级分组 | 最佳外部候选但尚不可用 |
| Zenodo 18700580 最小包 | 管线试审 | CC BY 4.0、49.7 MB、图像/掩码齐全 | 混合来源无逐文件清单；连续段复用掩码；无负样本 | 仅 pilot |
| CWFID | 否决 | 植被与作物/杂草标注 | 近俯视、非导航线标签 | 不进入本项目证据链 |

逐项机器可读记录见 `data/source_registry.json`。

## 3. 已下载试审包的真实检查

官方记录：<https://zenodo.org/records/18700580>
审查时归档位于 `D:\DL_code\data\crop_row_perception\sources\zenodo_18700580\S1_File.zip`；完成 MD5 与解压审查后，已按用户要求删除压缩包。80 对解压样本保留在 `extracted\S1_File\`。

- 官方大小：49,731,314 字节；
- 官方与本地 MD5：`79d2b2327e2389da496be8a40d6db415`，一致；
- 80 张 JPG、80 张 PNG，文件名一一对应；
- 160 个文件全部可解码；图像和掩码尺寸全部一致；
- 分辨率分成 `320x240` 48 对、`3840x2160` 11 对、`1024x576` 21 对；
- 掩码像素体系并不统一，出现 `0/1/2`、`0/2`、`0/255`、`0/1` 四种集合；
- 没有完全重复的图像文件；
- 发现 4 组完全相同的掩码，覆盖文件 369–388 中的 20 张；
- 元数据说明包内混有作者在固安采集的玉米图像和公开 CRBD 子集，但没有逐文件来源、农田、日期、设备或序列清单。

这说明“文件能打开”只通过了最低技术门槛；它没有通过无泄漏分组和独立外部证据门槛。

## 4. 为什么不能直接随机拆分

相邻文件名、相同分辨率与重复掩码共同提示连续采集或同构样本风险。若按图片随机拆分，同一场景的近邻帧很可能同时进入训练和测试，内部指标会虚高。正确的最高泄漏单位至少应是：

```text
source_dataset / field / acquisition_date / camera / sequence
```

这些键当前拿不到，所以审查器将 `provenance_groups_known` 判为 false，而不是猜测分组。

## 5. 最终数据下载设计

“最终数据”不是一个压缩包，而是三个互不混用的角色：

1. ID 首选 CRDLD v2.1：<https://lcas.lincoln.ac.uk/nextcloud/index.php/s/Eip4nWbetxJQ6No>。只有作者确认数据许可和最高分组键后才下载到 `D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\`。
2. OOD 开发候选 CCRDNet：<https://zenodo.org/records/15194034>。下载目标为 `D:\DL_code\data\crop_row_perception\sources\ccrdnet_15194034\`，但它只检验手持视角变化，不能替代机器人外部测试。
3. 冻结外部首选 Bonn Multi-Crop：<https://github.com/Agricultural-Robotics-Bonn/visual-multi-crop-row-navigation>。README 要求邮件申请；只有拿到数据许可和文件级分组后才能冻结。

还必须另建自有负样本集，覆盖地头、无行、非农田、严重模糊、过曝/欠曝与完全遮挡。负样本不能从支持帧简单裁剪伪造。

## 6. 解除阻断的最小条件

- CRDLD 作者或随包文件明确许可，并提供至少田块/日期/序列级分组；
- Bonn 数据获得实际访问、数据许可和独立来源清单；
- 建立合法、独立的应拒绝负样本来源；
- 下载后重新运行配对、解码、尺寸、标签值、重复/近重复和跨角色重叠审查；
- 只有审查器返回 `PASS` 才进入下一门。

## 7. 可复现命令

```powershell
D:\conda\envs\forest-species\python.exe 60_crop_row_data_viability\code\day60_data_audit.py `
  D:\DL_code\data\crop_row_perception\sources\zenodo_18700580\extracted\S1_File `
  --registry projects\04_crop_row_perception\data\source_registry.json
```

本次输出为 `80/80` 配对完整、零解码失败、4 组重复掩码，数据门禁为 `BLOCKED`。
