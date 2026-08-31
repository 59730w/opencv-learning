# Day59 开源作物行项目审查

审查日期：2026-08-31

## 1. 为什么先做开源审查

在开始新项目之前搜索相似实现是可行且必要的。它能帮助我们：

1. 确认问题是否已被合理定义；
2. 找到可复现的传统基线、评价指标和公开数据；
3. 提前识别真实田间的阴影、杂草、断行、弯曲和地头等失败条件；
4. 把时间用于复现、比较和有证据的改进，而不是重复造轮子。

但“参考开源项目”不等于复制代码。许可证、数据独立性、相机视角和评价协议都可能不同。Day59 只借鉴任务设计与算法思想，没有下载或复制第三方代码。

## 2. 候选项目比较

| 项目 | 已展示的价值 | 主要限制 | Day59 决策 |
|---|---|---|---|
| [petern3/crop_row_detection](https://github.com/petern3/crop_row_detection) | Python/OpenCV；使用 `2G-R-B` 强化植被并结合 Hough；README 报告约32.1 ms，同时坦率说明只检测到约50%的行 | 偏好三行、弯曲行和非绿色/过密作物困难；GPL-3.0 | 作为最接近当前能力的经典可解释基线思想；不复制代码，后续独立实现并重新评价 |
| [Agricultural-Robotics-Bonn/visual-multi-crop-row-navigation](https://github.com/Agricultural-Robotics-Bonn/visual-multi-crop-row-navigation) | IROS 2022、真实多作物行导航演示；提供多作物/仿真标注，并以图像底边位置和方向评价作物行 | ROS Melodic、旧版 OpenCV、系统依赖重；目标比12天学习项目更大 | 作为主要评价设计和多作物稳健性参考；不在 Day59 接入 ROS |
| [PRBonn/visual-crop-row-navigation](https://github.com/PRBonn/visual-crop-row-navigation) | C++/ROS、真实机器人与仿真验证、可在受限嵌入式控制器运行；MIT | 前后双相机、ROS Kinetic/Ubuntu 16.04，与当前 Windows 单目离线学习条件不一致 | 参考“视觉输出如何服务导航”和拒绝边界，不作为可直接运行基线 |
| [JunfengGaolab/CropRowDetection](https://github.com/JunfengGaolab/CropRowDetection) | CRDLD v2.1 描述2000张图、50种组合条件和行分割真值，覆盖阴影、作物大小、杂草、晴阴、断行、曲线与轮胎印 | Day59 页面未明确给出仓库/数据许可证；既有划分的最高泄漏单位仍需审计 | 列为 Day60 首要数据候选；许可证、下载可用性、序列/场景分组通过前不得作为正式数据 |
| [PatD123/Crop-Lane-Detect](https://github.com/PatD123/Crop-Lane-Detect) | IPM、DBSCAN、左右行聚类和滑动窗口时序平滑，方向与 Day63/65 相符 | README 承认 Hough 会产生错误线；Day59 页面未显示许可证和正式定量评价 | 只记录为后续受控增强候选，不能先于简单基线加入 |
| [WoodratTradeCo/crop-rows-detection](https://github.com/WoodratTradeCo/crop-rows-detection) | YOLOv5 ROI 检测；README 报告640x360约25 ms、平均角误差1.88° | 1500张训练图未全部公开；GPL-3.0；需要自建数据才能验证迁移 | 作为未来深度学习对照，不作为 Day61 的首个基线，也不接受其指标作为我们的证据 |
| [EarlJr53/row-crop-detection](https://github.com/EarlJr53/row-crop-detection) | 教学项目采用“先做简单方法，再比较复杂方法”的合理范围；列出消失点、曲线、光照和背景误检等改进方向 | 作者明确没有利用 CRBD 做正式基准，展示样例偏成功案例 | 参考项目学习节奏，不参考效果结论 |

## 3. 本项目的参考层次

### 第一层：必须先复现的可解释基线

独立实现以下流水线，不复制 GPL 项目代码：

```text
RGB 图像 → 植被指数/颜色阈值 → 形态学清理 → ROI → 行候选 → 几何拟合
```

这条线能复用 Day1-Day15 的 OpenCV 基础，也能让每一步失败都可观察。

### 第二层：通过失败证据再添加的增强

- 透视变换/IPM；
- 连通区域或 DBSCAN 抑制杂草离群点；
- 消失点/平行结构约束；
- 时序平滑与置信度；
- 深度学习分割或 ROI 检测对照。

这些增强只能由 Day67 的失败分组或前序证据触发，不能一次全部堆入。

### 第三层：当前不做

- ROS 闭环控制；
- 真实机器人导航；
- 多传感器融合；
- 直接复用别人的模型指标作为自己的项目成果。

## 4. 对今后项目的固定做法

以后开始实战项目时，在写模型或完整管线之前增加“开源与论文基线审查”：

1. 搜索3到7个任务真正相似的项目；
2. 检查代码、数据、许可证、环境、指标和失败说明；
3. 选一个最小可解释基线和一个较强对照；
4. 记录可借鉴内容、不可比较之处和许可证边界；
5. 独立复现并在自己的冻结协议下评价；
6. 只有新实验超过基线或改善明确失败组，才称为优化。

这会作为方法原则使用，但不会为了“参考更多”无限扩大检索和依赖。
