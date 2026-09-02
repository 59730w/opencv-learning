# Day61：颜色与光照处理

## 1. 今天要解决什么

农业机器人获得的是 BGR 图像。Day61 不直接求作物行，而是先生成“可能属于绿色植被”的候选掩码，为 Day62 的形态学与区域分析提供输入。

白色表示颜色上像植被，黑色表示当前颜色证据不足。这里有两个不能混淆的边界：

1. 颜色可以区分绿色与非绿色，却不能可靠区分绿色作物和绿色杂草；
2. CRDLD 标签是作物行中心线，不是植被区域掩码，因此本课指标是中心线代理指标，不是 IoU、Dice、精确率或误检率。

## 2. 三个版本是怎样演进的

### 第一版：8 张困难图上的 HSV、Lab、ExG+Otsu

第一版完成了颜色空间学习，但实验范围太小，参数也没有独立选参与验证。图 417 几乎没有明显绿色植被，逐图 ExG+Otsu 却激活约 54.9% 像素，暴露了根本问题：Otsu 每张图都想切成两类，即使画面没有可信前景也会强行制造一类。

### 第二版：固定阈值、互斥分区与风险拒绝

第二版改用固定 ExG，建立选参、审查、诊断和验证分区，并冻结 HSV 主方法：

```text
20 <= H <= 95
S >= 15
V >= 25
```

它在 64 张验证图中接受 61 张，将 3 张绿色覆盖过高的图标记为风险帧。接受图的平均 gap 为 0.277、P10 为 0.130、最大背景激活为 0.471。

这已经比第一版可靠，但仍有两个遗憾：

- 未使用风险判断时，最大背景激活仍为 0.582；
- 它主要在较小的训练开发子集上验证，尚未充分利用 1,250 张训练开发图与 248 张审计后的 validation-development 图。

### 第三版：有界 Gray-World + 冻结 HSV

第三版先对整张图做有界 Gray-World 光照校正，再执行冻结 HSV。它在开发压力检查和此前从未用于 Day61 的 248 张 validation-development 图上都稳定优于第二版，因此成为 Day62 的新主输入。

## 3. HSV、Lab 和 ExG 的作用

### 3.1 HSV

- H：色相，表示颜色偏绿、黄或蓝；
- S：饱和度，表示颜色鲜艳程度；
- V：亮度。

OpenCV 的 H 范围是 0–179，不是角度制的 0–360。HSV 把亮度单独放进 V，但不代表它对阴影和色偏免疫。

### 3.2 Lab

- L：明暗；
- a：大致从绿到红；
- b：大致从蓝到黄。

Lab 往往保留更多绿色区域，但也会增加背景激活。因此它保留为偏宽松的教学对照，没有成为最终主方法。

### 3.3 归一化 ExG

```text
r = R / (R + G + B)
g = G / (R + G + B)
b = B / (R + G + B)
ExG = 2g - r - b
```

归一化能减弱整体亮度变化。固定阈值 `ExG >= 0.03` 允许空植被图输出接近全黑，因此比逐图 Otsu 更符合“没有证据就不激活”的原则。

简单的 HSV/ExG/Lab 双门槛融合也经过开发实验，但它损失了过多行附近植被，未稳定超过 HSV。这是保留的负结果。

## 4. Gray-World 为什么可能改善光照

一张图受到光源颜色或相机白平衡影响时，B、G、R 三通道可能整体偏向某个颜色。Gray-World 假设一幅自然图像在足够多像素上的平均颜色应接近中性灰，据此调整通道增益：

```text
mean = (mean_B + mean_G + mean_R) / 3
gain_B = mean / mean_B
gain_G = mean / mean_G
gain_R = mean / mean_R
corrected_channel = channel * gain
```

为了避免某个通道均值过小时被无限放大，本课把每个增益限制在：

```text
0.5 <= gain <= 2.0
```

校正后的冻结 HSV 条件是：

```text
20 <= H <= 90
S >= 10
V >= 25
```

它比第二版略微收窄 H 上限，同时降低 S 下限，以便在校正色偏后保留低饱和度植被。

Gray-World 不是普遍真理。若画面几乎被一种真实颜色完全占据，“平均应为灰色”的假设可能不成立。因此必须通过大量开发样本和未见验证样本检查，不能只展示一张校正后更好看的图。

## 5. 指标怎样理解

先把中心线标签膨胀成窄邻域，再计算：

- `line_neighborhood_support`：行邻域内被激活的比例；
- `off_line_activation`：行邻域外被激活的比例；
- `candidate_fraction`：全图候选比例；
- `gap = support - off_line`：候选是否更集中于行附近；
- `P10 gap`：最差 10% 图像的区分度；
- `max off-line`：单张最坏背景激活。

全图设为白色会得到很高 support，却没有实际价值。因此鲁棒性评分既奖励平均 gap 和 P10，也惩罚平均及最坏背景激活。

## 6. 第三版怎样避免只对少量图片有效

### 6.1 训练开发源内的方案选择

1,250 张 CRDLD train-development 图被确定性划为：

| 用途 | 数量 |
| --- | ---: |
| 参数搜索 | 256 |
| 方法审查 | 256 |
| 压力检查 | 738 |

开发时比较了：

- 第二版固定 HSV；
- Gray-World 后沿用旧 HSV；
- Gray-World 后的小网格 HSV；
- HSV 与固定 ExG 的交集；
- HSV、Lab 和 ExG 的双证据融合。

最终冻结 Gray-World + HSV `20–90/10/25`。压力检查结果：

| 方法 | support | off-line | gap | P10 | max off-line | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 第二版 HSV | 0.516 | 0.255 | 0.261 | 0.106 | 0.688 | 0.180 |
| 第三版 GW+HSV | 0.517 | 0.248 | 0.268 | 0.129 | **0.487** | **0.294** |

这表明改进不是靠牺牲大量作物支持换来的：support 略升，平均 gap、P10 和最坏背景激活同时改善。

### 6.2 未见 validation-development 验证

参数冻结后，程序首次读取审计清单中的 248 张 validation-development 图。重复图 49、167 已由数据门禁清单排除。

在读取前预先写入两类门槛：

- 绝对门槛：mean gap ≥ 0.20、P10 ≥ 0.08、max off-line ≤ 0.55；
- 相对门槛：分数至少提高 0.05、最坏背景至少降低 0.10、平均 gap 最多下降 0.01、P10 不下降。

任何一项失败，第三版都不能替换第二版。

## 7. 248 张最终开发验证结果

| 指标 | 第二版 HSV | 第三版 GW+HSV | 变化 |
| --- | ---: | ---: | ---: |
| line support | **0.443** | 0.438 | -0.005 |
| off-line activation | 0.190 | **0.180** | -0.010 |
| mean gap | 0.254 | **0.258** | +0.004 |
| P10 gap | 0.146 | **0.171** | +0.025 |
| max off-line | 0.708 | **0.390** | -0.319 |
| robustness score | 0.179 | **0.293** | +0.114 |

结论：所有预设绝对与相对门槛通过，第三版正式替换第二版作为 Day62 输入。

这次不需要先排除高覆盖图再计算：248 张全部参与汇总。最明显的改进出现在 180、184、185、186 等受明显色偏影响的图上；旧 HSV 大面积激活土壤，新方法保留作物并压制背景。

代价是平均行邻域支持降低约 0.0053，即约 0.53 个百分点。这个损失很小，同时换来了最坏背景激活降低约 31.9 个百分点和更好的尾部表现，权衡合理。

## 8. 第三版仍然不能解决什么

1. 作物与杂草都为绿色时，颜色仍无法区分语义；
2. 掩码仍有孔洞、碎片和小噪点，这是 Day62 的任务；
3. 田边浓密绿色植被需要 Day63 的 ROI 和行几何约束；
4. Gray-World 在极端单色占满画面时可能失效；
5. CRDLD 的最高泄漏分组信息仍未知；
6. RowDetr 冻结外部集没有访问，因此没有外部泛化结论；
7. 当前指标基于中心线代理，不是植被分割真值指标。

## 9. 代码结构

主程序：`code/day61_color_illumination.py`

核心函数：

- `gray_world_balance`：有界 Gray-World 校正；
- `segment_grayworld_hsv`：第三版主候选；
- `segment_hsv`：第二版基线；
- `segment_lab`、`segment_exg_fixed`：教学对照；
- `segment_exg_otsu`：明确否决的失败基线；
- `read_manifest_stems`：只读取审计清单声明的验证项；
- `compute_line_proxy_metrics`、`proxy_robustness_score`：代理指标与尾部评分；
- `run_v4_study`：开发汇总、未见验证、门槛判断与可视化。

## 10. 怎样复现第三版

```powershell
D:\conda\envs\forest-species\python.exe `
  61_crop_row_color_illumination\code\day61_color_illumination.py `
  --train-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\train_data `
  --validation-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\validation_data `
  --validation-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_validation_development_manifest.jsonl `
  --output-dir D:\DL_code\data\crop_row_perception\day61_color_illumination
```

主要输出：

```text
comparison_contact_sheet_v4.jpg  第二版与第三版视觉对比
proxy_metrics_v4.csv             248 张逐图指标
day61_results_v4.json            配置、门槛和汇总结果
```

原图和包含原图的结果不提交 Git，因为 CRDLD 许可仍未明确。

## 11. 自动测试

测试包括：

- Gray-World 对中性灰保持不变，并减小全局通道色偏；
- 输入图像不被原地修改；
- GW+HSV 输出同尺寸二值 `uint8` 掩码；
- 验证清单只读取正确 evidence role 的唯一 item ID；
- 固定 ExG 不会在中性空场景强行二分；
- 四个旧实验分区确定且互斥；
- 代理指标与鲁棒性评分能惩罚背景和差尾部。

运行：

```powershell
D:\conda\envs\forest-species\python.exe -m pytest `
  61_crop_row_color_illumination\tests\test_day61_color_illumination.py -q
```

## 12. 今天真正应该学会什么

1. 光照问题不只等于亮度问题，全局色偏会直接改变 HSV 判断；
2. 预处理方法也有假设，Gray-World 必须验证，不能因名字听起来稳健就采用；
3. 方法应在大开发集上选择，在未见清单上按预设门槛一次验证；
4. 平均指标之外，P10 和单张最坏背景更接近机器人系统真正担心的失败；
5. 负结果同样重要：简单颜色融合没有胜出，因此没有进入管线；
6. 同源验证改善不能冒充外部泛化。

## 13. Day61 最终状态

```text
DAY61_LESSON_COMPLETE
GRAYWORLD_HSV_BASELINE_ACCEPTED_FOR_DAY62
SAME_SOURCE_DEVELOPMENT_ONLY
EXTERNAL_GENERALIZATION_BLOCKED
```
