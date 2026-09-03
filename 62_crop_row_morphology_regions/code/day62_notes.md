# Day62：形态学与区域分析

## 今天做成了什么

Day61 的任务是从照片里找出“可能是绿色植物”的白色像素。它得到的掩码像一张黑白图：白色可能是植物，黑色表示颜色证据不足。

Day62 的任务是整理这张黑白图：

1. 去掉零散的小白点；
2. 填补植物区域中的小裂缝；
3. 把相连的白色像素划成一个个区域；
4. 保存面积、位置、外接框等信息，交给 Day63 拟合作物行。

第二版最终冻结的方法是：

```text
Day61 Gray-World + HSV 掩码
        ↓
3×3 opening：去除细小噪点
        ↓
5×7 纵向 closing：优先连接上下方向的短裂缝
        ↓
透视感知面积过滤：远处允许更小的区域，近处要求更大
        ↓
Day63 的输入
```

结论：**Day62 已达到优秀的学习阶段成果，可以进入 Day63。**这里的“优秀”表示方法完整、实验严格、输出适合继续学习，不表示已经能够可靠控制真实农业机器人。

## 1. 为什么第一版合格，但不算优秀

第一版使用：

```text
3×3 opening + 5×5 closing + 全图统一面积阈值
```

它在 248 张同源验证图上把平均连通域数从 `369.3` 降到 `43.4`，同时 mean gap 和 P10 都提高，因此已经能供 Day63 使用。

问题在于全图都使用相同的面积下限。透视画面中，远处的植物本来就比近处小；统一阈值可能把远处真实植物和噪点一起删除。此外，第一版只检查总体碎片和中心线附近激活，没有直接检查白色候选是否沿画面纵向连续。

所以第一版的评价是：

```text
去噪效果：很好
中心线代理：通过
远处植物保护：不够细致
Day63 几何就绪性：可用，但仍可改善
```

## 2. 第二版改进了什么

### 2.1 方向性 closing

第一版的 `5×5` 核向上下左右同等扩张。第二版改为 `5×7`：宽 5 像素、高 7 像素。

它更愿意连接上下方向的短裂缝，而较少向左右扩张。作物行在前视画面中通常从近处向远处延伸，因此这个方向更适合为 Day63 保留纵向结构。

这不是在 Day62 直接拟合直线；它只是让候选区域更适合下一天做几何分析。

### 2.2 透视感知面积过滤

第一版在 640×480 图像上统一删除不足约 62 像素的连通域。

第二版仍把 62 像素作为图像底部附近的标准，但顶部阈值只取它的 40%，中间位置线性过渡：

```text
位置越靠上 → 植物看起来越小 → 面积要求越低
位置越靠下 → 植物看起来越大 → 面积要求越高
```

公式为：

```text
scale = 0.4 + 0.6 × y_norm
min_area = ceil(图像面积 × 0.00020 × scale)
```

其中 `y_norm=0` 表示图像顶部，`y_norm=1` 表示底部。

## 3. opening、closing、连通域和轮廓是什么

### Opening

先腐蚀再膨胀。可以理解为先把白色区域削薄，小白点会消失；再把剩余主体恢复一些。

主要作用：去掉孤立小噪点。

风险：细小真实叶片也可能被删除。

### Closing

先膨胀再腐蚀。可以理解为先让相邻白色区域靠近并连接，再恢复大致尺寸。

主要作用：补小孔、接短裂缝。

风险：本来无关的植物或杂草可能被错误连接。

### 连通域

互相连接的白色像素组成一个连通域。OpenCV 可以为每个连通域返回：

- 像素面积；
- 外接框；
- 质心位置。

本课真正执行的面积过滤使用连通域的像素面积。

### 轮廓

轮廓是区域的外边界。OpenCV 可以计算轮廓面积、周长和外接框。本课将轮廓画在原图上，帮助检查哪些植物区域被保留下来。

注意：连通域面积是白色像素数量，轮廓面积是边界围成的几何面积，两者数值并不相同。

## 4. 怎样保证不是在少数图片上碰巧有效

第二版只使用 1,250 张 CRDLD train-development 图像选择方法，并把它们确定性分成 5 折，每折 250 张。

每个候选都必须在至少 4 折中满足保护门槛。结果如下：

| 候选 | 五折通过数 |
|---|---:|
| 3×7 closing，统一面积阈值 | 0/5 |
| 3×9 closing，统一面积阈值 | 1/5 |
| 5×7 closing，统一面积阈值 | 0/5 |
| 3×7 closing，透视阈值 | **5/5** |
| 3×9 closing，透视阈值 | 1/5 |
| 5×7 closing，透视阈值 | **5/5** |

在两个稳定候选中，综合中心线代理、纵向支持、尾部表现和前景保留后，冻结 `5×7 closing + 透视阈值`。

在此之前还测试过只修改透视面积阈值、不修改 closing 方向的方案。它虽然少删除了一些前景，但纵向连续性改善太小，没有通过升级门槛，因此作为负结果保留，没有通过放宽门槛强行采用。

## 5. 第二版必须通过哪些门槛

与第一版相比，第二版必须同时做到：

- mean support 最多下降 0.005；
- mean gap 最多下降 0.003；
- P10 gap 最多下降 0.005；
- 最坏背景激活最多增加 0.01；
- 平均连通域仍至少减少 75%；
- 平均连通域数不超过第一版的 1.5 倍；
- 删除的原始前景至少比第一版少 0.002；
- 纵向带支持和最长连续覆盖不能明显下降；
- 纵向几何指标或前景保留至少有一项达到预设改善量。

如果没有候选满足条件，程序会继续使用第一版。

## 6. 248 张 validation-development 的确认结果

这 248 张图在第一版已经看过，所以第二版只能把它们叫作“已复用的确认集”，不能再叫新的独立验证集。第二版参数完全由 1,250 张 train-development 的五折结果选择，确认集没有用于挑参数。

| 指标 | 第一版 | 第二版 | 解释 |
|---|---:|---:|---|
| line support | 0.4429 | **0.4476** | 行附近保留更多候选 |
| off-line activation | **0.1764** | 0.1789 | 背景激活略增 |
| mean gap | 0.2665 | **0.2687** | 总体区分度略升 |
| P10 gap | 0.1761 | **0.1780** | 较差样本略升 |
| max off-line | **0.3970** | 0.4023 | 最坏背景略增，但在门槛内 |
| robustness score | 0.3017 | **0.3043** | 综合分数略升 |
| 平均连通域数 | **43.4** | 47.3 | 多保留约 3.9 个区域 |
| 删除原始前景 | 5.05% | **4.51%** | 少删除约 0.54 个百分点 |
| 纵向带平均支持 | 0.4248 | **0.4286** | 更适合后续纵向几何分析 |
| 较差10%纵向支持 | 0.3065 | **0.3099** | 尾部略改善 |
| 最长连续覆盖 | 0.8775 | **0.8785** | 小幅改善 |

第二版不是巨大跃升，而是受约束的小幅升级：它多保留一些远处和纵向植被信息，代价是背景激活和区域数量略增。全部变化都在预注册保护范围内。

## 7. 这些结果达到优秀了吗

分两种含义回答：

### 作为 Day62 学习成果：优秀

- 完整学习了 opening、closing、结构元素形状、连通域、轮廓和面积过滤；
- 第一版问题有明确证据，不是凭感觉优化；
- 第二版使用五折稳定性选择，不靠单一切分；
- 失败的纯透视方案被保留；
- 第二版通过预注册门槛；
- 输出已经包含 Day63 需要的干净候选区域、轮廓和区域统计。

### 作为真实农业机器人算法：还不能称优秀

- CRDLD 标签只是中心线，不是植被区域真值；
- 当前没有目标域拒识负样本；
- 最高泄漏分组未知；
- CRDLD test 和 RowDetr 外部集尚未访问；
- 还没有完成 Day63 几何、Day65 时序和 Day66 完整管线。

因此正确结论是：

```text
Day62 学习任务：优秀，可以进入 Day63
真实系统可靠性：尚无足够证据
外部泛化：仍然 BLOCKED
```

## 8. 能否满足后面几天的学习

| 后续任务 | Day62 是否满足 | 原因 |
|---|---|---|
| Day63 作物行几何提取 | **满足** | 已提供较连续的二值区域、连通域和轮廓 |
| Day64 偏移和方向角 | **间接满足** | 先由 Day63 把区域变成行模型 |
| Day65 视频时序稳定 | **间接满足** | 可逐帧产生稳定格式的候选区域 |
| Day66 完整管线 Pilot | **满足前置条件** | 颜色和区域处理可以串联 |
| Day67 失败分析 | **满足前置条件** | 已保存逐图指标和视觉对照 |
| Day68 受控改进 | **满足前置条件** | 有冻结基线和明确失败边界 |
| Day69 冻结外部测试 | **尚不能执行** | 必须先冻结 Day63–68 的完整管线 |

## 9. 代码和结果在哪里

主程序：

```text
62_crop_row_morphology_regions/code/day62_morphology_regions.py
```

主要函数：

- `apply_morphology`：执行普通或方向性开闭运算；
- `filter_components_by_area`：第一版统一面积过滤；
- `filter_components_perspective`：第二版透视感知面积过滤；
- `component_records`：连通域统计；
- `contour_records`：轮廓统计；
- `vertical_line_support_metrics`：纵向几何就绪指标；
- `v2_acceptance_checks`：第二版升级门槛；
- `select_v2_from_fold_summaries`：五折稳定选择；
- `run_v2_study`：完整第二版实验。

本地结果：

```text
D:/DL_code/data/crop_row_perception/day62_morphology_regions/
```

主要文件：

```text
day62_results.json                              第一版结果
day62_results_v2_perspective_only_rejected.json 纯透视失败结果
day62_results_v2.json                           最终第二版结果
proxy_region_metrics_v2.csv                     五折与确认集逐图指标
comparison_contact_sheet_v2.jpg                 第一版/第二版视觉对比
```

由于 CRDLD 许可仍未明确，包含原图的结果不上传 Git。

## 10. 复现命令

```powershell
D:\conda\envs\forest-species\python.exe `
  62_crop_row_morphology_regions\code\day62_morphology_regions.py `
  --study v2 `
  --train-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\train_data `
  --train-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_train_development_manifest.jsonl `
  --validation-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\validation_data `
  --validation-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_validation_development_manifest.jsonl `
  --output-dir D:\DL_code\data\crop_row_perception\day62_morphology_regions
```

## 11. Day62 最终状态

```text
DAY62_V2_LESSON_COMPLETE
DIRECTIONAL_PERSPECTIVE_MORPHOLOGY_ACCEPTED_FOR_DAY63
TRAIN_ONLY_FIVE_FOLD_SELECTION
REUSED_VALIDATION_CONFIRMATION_ONLY
SAME_SOURCE_INTERNAL_BENCHMARK_NOT_ACCESSED
FROZEN_EXTERNAL_NOT_ACCESSED
EXTERNAL_GENERALIZATION_BLOCKED
```
