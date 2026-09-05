# Day63：多作物行几何重学——从单行基线到多行中心线网络

> 2026-09-05 原位重学说明：第1～17节与附录A～F保留第一版和单行第二版历史，
> 不再代表当前最终方法。当前结论、指标和新项目交接以第18～24节为准。

## 1. 今天完成了什么

Day63 最初把 Day62 冻结的二值候选区域转换为一条可解释的中央作物行几何；
这条旧链路现在只作为单行基线保留：

```text
RGB图像
  → Day61冻结的有界Gray-World + HSV
  → Day62冻结的3×3 opening + 5×7纵向closing + 透视感知区域过滤
  → 归一化ROI
  → Hough基线 / 透视约束整线支持搜索
  → 中央作物行近点、远点、方向代理、置信度和状态
```

今天没有重新调整 Day61 的颜色阈值，也没有重新调整 Day62 的形态学参数。

最终选择的方法是：

```text
support_sigma9_center10_angle20
```

它使用完整直线假设搜索，而不是从许多局部 Hough 线段中直接挑一条。训练开发集的五个互斥折全部通过预设的平均位置误差、平均方向误差和有效召回门槛。

## 2. 今天没有做什么

- 没有访问 CRDLD `test_data` 同源内部基准；
- 没有访问 RowDetr 冻结外部集；
- 没有目标域负样本，因此没有验证真正的拒识能力；
- 没有相机内参、外参或地面单应性标定；
- 没有把图像平面偏移换算成米；
- 没有实现机器人控制；
- 没有上传数据、原图、生成结果或模型。

所以 Day63 的证据只能支持“同源正样本几何开发”，不能支持外部泛化或真实农业机器人可靠性。

## 3. 输入为什么必须冻结

如果在几何结果不好时同时修改颜色阈值、形态学核和直线方法，就无法判断改进来自哪一层。Day63 固定使用：

```python
DAY62_FROZEN_CONFIG = {
    "name": "directional_close5x7_perspective40",
    "order": "open_close",
    "open_kernel": 3,
    "close_kernel": [5, 7],
    "min_area_fraction": 0.0002,
    "perspective_top_scale": 0.4,
    "perspective_exponent": 1.0,
}
```

这让今天的比较只回答一个问题：在相同候选区域上，哪一种几何提取方法更可靠？

## 4. 几何目标怎样定义

CRDLD 标签中有多条作物行中心线。今天必须先固定“中央目标行”，否则算法可能通过看真值临时选择最有利的一条线。

目标定义为：

1. 在近端评价横线 `y_norm = 0.90` 观察所有标注行；
2. 选择近端交点最接近图像中心 `x_norm = 0.50` 的标注行；
3. 从近端向远端连续跟踪这条标注行；
4. 用 Huber 鲁棒直线拟合得到参考近点和远点；
5. 预测阶段不能读取标签。

远端评价横线固定为：

```text
y_far_norm = 0.40
```

方向代理沿用 Day59 的符号约定：

```text
heading_deg = degrees(
    atan2(x_far_norm - x_near_norm,
          y_near_norm - y_far_norm)
)
```

正值表示直线从近端向远端朝图像右侧，负值表示朝图像左侧。Day64 会把近点、远点进一步组织为正式的偏移与方向输出。

## 5. ROI：为什么先排除图像顶部

图像顶部常包含天空、树带、地平线和分辨率很低的远端植被。这些区域对近端导航几何帮助有限，却容易产生大量错误线段。

今天使用归一化 ROI：

```text
y_norm >= 0.25
```

归一化坐标而不是固定像素值，使相同规则能迁移到不同分辨率。测试同时在 `100×200` 和 `200×400` 掩码上验证了这个性质。

## 6. 透视变换学到了什么

代码实现了一个梯形到矩形的示范性单应变换：

```python
matrix = cv2.getPerspectiveTransform(source, destination)
warped = cv2.warpPerspective(mask, matrix, (width, height))
```

它可以直观看到透视收敛的作物行被拉直后的形态，也验证了正向矩阵和逆矩阵能够恢复测试点。

但这个变换没有相机标定和真实地面控制点，所以只能叫“未标定透视示范”，不能叫真实鸟瞰图，更不能用于米制距离测量。

最终几何方法没有在变换后的图像中直接测物理量，而是在原图归一化坐标中使用一个弱透视先验。

## 7. 为什么普通 Hough 没有成为最终方法

Hough 基线使用：

```python
cv2.HoughLinesP(...)
```

它会寻找局部共线前景并产生多条线段，然后根据近端交点是否靠近图像中心挑选候选。

它的优点是：

- 原理直观；
- 可以快速产生可视化线段；
- 适合当作最低限度的几何基线。

它在本数据上的问题是：

- 一株植物的左右边缘可能产生不同线段；
- 石块、阴影和叶片边缘也可能形成长直线；
- 局部线段方向不等于完整作物行方向；
- 多条作物行同时存在时，靠近中心的局部线段未必属于中央目标行。

训练五折中，Hough 的有效率只有约 `51%–58%`，平均方向误差约 `15°–19°`。因此它只保留为学习基线，没有被选为 Day64 输入。

## 8. 最终方法：透视约束整线支持搜索

最终方法不是先找局部线段，而是直接枚举一组完整的近点—远点直线假设。

### 8.1 候选范围

```text
near_y_norm = 0.90
far_y_norm  = 0.40
near_x_norm ∈ [0.18, 0.82]
far_x_norm  ∈ [0.34, 0.66]
```

远端范围更窄，表达作物行通常向图像中央消失区域收敛的弱透视先验。

### 8.2 前景支持

先对 Day62 掩码进行横向高斯平滑：

```text
sigma_x = 9
sigma_y = 2
```

较宽的横向平滑允许直线穿过叶片宽度、局部断行和轻微中心偏差，而不是要求每个采样点都正好落在白色像素上。

每条候选线在远近评价线之间均匀采样 36 次，记录：

- `mean_support`：整条线平均前景支持；
- `p20_support`：支持度第20百分位，用来暴露断裂区域。

### 8.3 评分

候选分数由四部分构成：

```text
score = mean_support
      + 0.35 × p20_support
      - 1.0 × |near_x_norm - 0.5|
      - 0.2 × |heading_deg| / 45
```

前两项奖励完整作物行支持；第三项符合“中央目标行”的任务定义；第四项是弱方向正则，防止石块或叶片边缘产生极端斜线。

这些先验不能证明真实世界正确，只能说明候选规则是固定、可解释且不读取当前图片真值的。

### 8.4 有效、降级和拒绝

置信度定义为：

```text
confidence = 0.75 × mean_support + 0.25 × p20_support
```

它只是可记录的支持分数，不是概率。

状态规则为：

- ROI 没有前景或平均支持极低：`reject`；
- `confidence < 0.12`：`degraded`；
- 近端交点不在中央半幅 `[0.25, 0.75]`：`degraded`；
- 其余情况：`valid`。

把搜索边界上的结果降级非常重要。它表示算法很可能被相邻作物行或图像边缘强植被吸引，不能继续假装自己找到了中央目标行。

由于目前没有目标域负样本，这些状态仍是正样本几何质量状态，不是经过验证的安全拒识器。

## 9. 测试驱动过程

先创建测试并确认因模块不存在而失败：

```text
ModuleNotFoundError: No module named 'day63_crop_row_geometry'
```

随后逐步实现并验证：

- Day62 冻结配置没有被更改；
- ROI 对分辨率无关；
- 非二值输入和非法 ROI 会报错；
- 近点、远点归一化正确；
- 合成倾斜作物行能被恢复；
- 存在干扰线和噪声时仍选择中央行；
- 空掩码不会强制输出直线；
- 断续但整体有支持的行不会只因 `p20=0` 被错误降级；
- 搜索边界候选会被降级；
- Hough 基线能处理干净合成直线；
- 标签中央行选择规则固定；
- 位置和方向误差符号正确；
- 五折必须全部通过；
- 候选选择不读取确认集指标；
- 透视矩阵与逆矩阵匹配；
- 小型端到端研究会写出 JSON、CSV 和对比图，并保留证据边界。

第一版完成时的 Day63 测试为：

```text
17 passed
```

## 10. 数据与选择协议

### 10.1 训练选择

1,250 张 `train_development` 图片通过固定 SHA-256 排序分成五个互斥折，每折 250 张。

比较项目包括：

- 图像中央竖线：完全不读取图像的 sanity baseline；
- `HoughLinesP`：普通 Hough 基线；
- 四组不同平滑宽度和正则强度的整线支持搜索。

候选必须在五折中同时满足：

```text
valid_fraction >= 0.85
bottom_position_mae_norm <= 0.05
heading_mae_deg <= 5.0
```

最终四个整线候选都通过，按五折平均位置误差优先、方向误差次优选择：

```text
support_sigma9_center10_angle20
```

### 10.2 复用开发集

248 张 `validation_development` 在 Day62 已经使用，Day63 的前置筛选也访问过它。它随后暴露了搜索边界会选错相邻行的问题，因此最终状态规则把中央半幅之外的候选降级。

这意味着这 248 张图片现在是明确参与 Day63 开发的同源数据，不能再称为“未触碰确认集”。Day63 最终没有独立确认结果。

真正未触碰的仍然是：

- CRDLD 同源内部基准；
- RowDetr 冻结外部正样本。

## 11. 最终定量结果

### 11.1 训练开发五折

| 指标 | 五折平均 | 最差折 | 门槛 |
|---|---:|---:|---:|
| valid fraction | 0.8632 | 0.8560 | ≥ 0.85 |
| available fraction | 0.9584 | 0.9400 | 仅记录 |
| 近端位置 MAE | 0.0280 | 0.0289 | ≤ 0.05 |
| 方向 MAE | 3.532° | 3.630° | ≤ 5.0° |
| 位置 P90 | 0.0529 | 0.0593 | 尾部记录 |
| 方向 P90 | 7.518° | 7.646° | 尾部记录 |
| 单帧全链路中位耗时 | 38.09 ms | 38.34 ms | ≤ 50 ms |

结果：五折全部通过，训练选择门为 `PASS`。

### 11.2 已参与开发的248张同源数据

| 方法 | valid fraction | 位置 MAE | 方向 MAE | 同时过双阈值比例 | 中位耗时 |
|---|---:|---:|---:|---:|---:|
| 中央竖线 | 1.0000 | 0.0809 | 2.922° | 0.2823 | 29.44 ms |
| Hough | 0.4476 | 0.0723 | 14.940° | 0.2782 | 55.97 ms |
| 最终整线搜索 | 0.8589 | 0.0320 | 4.260° | 0.6089 | 40.06 ms |

最终方法达到四个合同均值/召回/耗时阈值，并明显优于 Hough。它也把中央竖线的位置误差从 `0.0809` 降到 `0.0320`，说明结果不是简单依靠“永远预测图像中心”。

但 `0.6089` 表示只有约60.9%的全部帧同时满足位置和方向两个逐帧阈值，不能据此宣称每张图片都可靠。

## 12. 尾部和最差案例

最终方法在这 248 张已参与开发图片上的尾部为：

```text
position P90 = 0.0615
heading P90  = 9.615°
position max = 0.3610
heading max  = 21.983°
```

这些指标明显差于均值，说明少量困难帧仍可能：

- 选到中央行旁边的相邻作物行；
- 被一侧强植被吸引；
- 在杂草或碎石很多时得到错误方向；
- 把局部直线当成完整作物行；
- 在弯曲作物行上受到直线模型限制。

对比图故意包含最佳、中位、低置信度、最大位置误差和最大方向误差帧，而不是只挑成功案例。

这些失败应在 Day67 分类，在 Day68 只选择一个主要原因进行受控改进。今天不能继续无限调参，否则会把 Day63 开发数据用尽。

## 13. 为什么中间失败轮次必须保留

本地输出保存了两个中间结果：

```text
day63_results_preliminary_rejected.json
day63_results_round2_train_gate_only.json
```

第一轮把 `p20_support` 当成硬有效条件，导致断续但整体存在的正常行大量降级，训练折 valid fraction 只有约 `0.51–0.59`。

第二轮取消单独的 `p20` 硬门后训练五折通过，但已复用开发数据暴露了搜索边界候选会高误差选到相邻行。

最终轮加入中央半幅有效性检查，并把误差指标严格计算在 `valid` 输出上，同时继续记录 `degraded` 与 `available_fraction`。

这三步属于同一个 Day63 受控开发流程，不是隐藏失败后只展示漂亮结果。最终方法的所有局限和开发数据使用历史都写进 JSON。

## 14. 代码中的主要函数

主程序：

```text
63_crop_row_geometry_extraction/code/day63_crop_row_geometry.py
```

关键函数：

- `apply_normalized_roi`：按归一化高度应用 ROI；
- `perspective_matrices`：生成示范性透视矩阵和逆矩阵；
- `hough_geometry`：普通概率 Hough 基线；
- `perspective_support_geometry`：最终整线支持搜索；
- `central_label_geometry`：固定规则提取中央参考行；
- `evaluate_prediction`：计算近端位置和方向误差；
- `split_geometry_folds`：构建确定性五折；
- `candidate_acceptance_checks`：检查五折绝对门槛；
- `select_train_only_candidate`：只使用训练折选择几何配置；
- `summarize_metric_rows`：汇总有效率、均值、P90、最大值和耗时；
- `run_day63_study`：执行完整研究并写出结果。

## 15. 本地结果

```text
D:/DL_code/data/crop_row_perception/day63_crop_row_geometry/
```

主要文件：

```text
day63_results.json                         第一版结构化结果
geometry_metrics.csv                      五折和同源开发逐图指标
geometry_contact_sheet.jpg                8张最佳/典型/失败对比
day63_results_preliminary_rejected.json    第一轮失败记录
day63_results_round2_train_gate_only.json  第二轮记录
```

对比图每行依次展示：

```text
原图 | Day62冻结掩码 | 未标定透视示范 | Hough与真值 | 第一版方法与真值
```

黄色为预测线，紫色为中央参考行。

## 16. 复现命令

```powershell
D:\conda\envs\forest-species\python.exe -X utf8 `
  63_crop_row_geometry_extraction\code\day63_crop_row_geometry.py `
  --train-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\train_data `
  --train-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_train_development_manifest.jsonl `
  --validation-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\validation_data `
  --validation-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_validation_development_manifest.jsonl `
  --output-dir D:\DL_code\data\crop_row_perception\day63_crop_row_geometry `
  --study v1 `
  --count 8
```

测试命令：

```powershell
D:\conda\envs\forest-species\python.exe -X utf8 -m pytest `
  63_crop_row_geometry_extraction\tests\test_day63_crop_row_geometry.py -q `
  --basetemp D:\opencv-learning\.tmp\day63-pytest
```

## 17. Day63 第一版评价（已被第二版替代）

### 作为 Day63 学习任务：优秀，可以进入 Day64

- ROI 使用归一化坐标；
- 实现并解释了未标定透视变换；
- Hough 作为真实基线被量化比较，而不是只演示 API；
- 最终方法对完整作物行评分，避免直接依赖局部线段；
- 五折全部通过；
- 平均位置、方向、有效召回和耗时达到合同门槛；
- 测试、逐图指标、尾部指标、可视化和失败轮次齐全；
- Day61/62 输入保持冻结；
- 内部测试和外部数据保持未访问。

### 作为真实农业机器人算法：还不能称优秀

- 248张数据已经参与开发，没有独立确认；
- P90 和最大误差仍高；
- 只有约60.9%的全部开发帧同时满足两个逐帧误差阈值；
- 缺少拒识负样本；
- 最高泄漏分组未知；
- 没有相机标定；
- 还没有视频时序、完整管线和冻结外部测试。

正确结论是：

```text
DAY63_LESSON_COMPLETE
TRAIN_FIVE_FOLD_GEOMETRY_GATE_PASS
SELECTED_SUPPORT_SIGMA9_CENTER10_ANGLE20
REUSED_VALIDATION_DEVELOPMENT_TOUCHED
NO_UNTOUCHED_CONFIRMATION_REMAINS
SAME_SOURCE_INTERNAL_BENCHMARK_NOT_ACCESSED
FROZEN_EXTERNAL_NOT_ACCESSED
REJECT_AWARE_EVALUATION_BLOCKED
READY_FOR_DAY64_MEASUREMENT_LESSON
REAL_ROBOT_RELIABILITY_NOT_ESTABLISHED
```

## 18. 为什么必须原位重学 Day63

Day59 合同已经从“找一条中央作物行”修订为“找出所有可评价作物行，再选择相机中心
左右相邻边界并计算行间走廊”。因此旧第二版虽然能准确回归一条中央参考行，却无法回答：

- 画面中到底有多少条可评价作物行；
- 哪两条是当前走廊左右边界；
- 中央作物行出现时为什么必须降级，而不能把它当作行驶中心；
- 多条线是否具有一致的透视方向。

本次没有新建重复的 Day63 目录，而是在原代码、测试和笔记中保留旧方法作为基线，加入
多行实现。Day61 颜色和 Day62 形态学参数均未重调。

## 19. 多行重学的失败轮次

重学不是一次得到好结果，中间负面结果均保存在本地：

1. 传统多带峰值、Hough消失点与规则格网投票：复用验证的召回约0.51～0.66，失败；
2. 一维多尺度投影网络：训练召回约0.90，复用验证召回约0.74，因丢失二维透视关系失败；
3. Tiny U-Net三折交叉拟合：总体召回0.932，但精确率0.726、方向MAE 6.303°，失败；
4. ResNet18整训：训练P/R接近0.99，复用验证召回仍约0.765，暴露训练与验证行密度偏移；
5. 最终方法：按数据分区进行三折、当前折标签排除的ResNet18中心线交叉拟合。

这里没有通过固定预测“五行/六行”来刷分。行数回归在训练到验证的测试中平均只预测4.44行，
而验证真值平均5.71行，证明固定行数或数据集先验不能作为真实多行检测方法。

## 20. 最终方法

输入为四通道：

```text
RGB三通道 + Day62冻结二值形态学掩码一通道
```

模型使用本机缓存的ImageNet ResNet18编码器和轻量U-Net解码器，在192×192分辨率输出
作物行中心线热图。解码器在固定 `y_norm=0.40` 水平带找峰值确定行数，再利用二维热图
拟合每条行的远点、近点与方向。输出按远端位置从左到右排序。

走廊只在以下条件同时满足时生成：

1. 相机中心左、右各存在一条相邻检测行；
2. 相机中心排除带内没有作物行；
3. 左右边界都至少得到三条水平带支持；
4. 走廊中心是左右边界的中线，不是任意一条作物行。

## 21. 交叉拟合协议与证据边界

训练开发1250张和已复用验证开发248张分别做三折。每张报告成绩的图片都由没有读取该图
标签的折模型预测：

- 训练开发折：从本地ImageNet ResNet18权重开始，在其余两折训练10轮；
- 验证开发折：从训练开发模型开始，只在其余两个验证折微调8轮；
- 当前验证折的标签不参与其模型拟合。

该协议降低了直接训练回看的偏差，但它不是未触碰确认：架构和解码阈值是在两套开发数据
已经被多次访问后确定的，而且CRDLD最高泄漏分组仍未知。因此这些成绩只能叫同源开发
out-of-fold证据，不能叫独立内部测试或外部泛化。

## 22. 最终多行结果

| 分区 | 行精确率 | 行召回率 | 匹配位置MAE | 匹配方向MAE | 多行几何门 |
|---|---:|---:|---:|---:|---|
| 训练开发 OOF，1250张 | 0.9552 | 0.9585 | 0.0101 | 1.652° | PASS |
| 复用验证开发 OOF，248张 | 0.9507 | 0.8178 | 0.0120 | 2.873° | PASS |
| 合计 OOF，1498张 | 0.9544 | 0.9298 | 0.0104 | 1.871° | PASS |

预设的四个Day63多行几何门为P≥0.80、R≥0.80、位置MAE≤0.05、方向MAE≤5°；
两个分区均全部通过。相比传统多带方法，最终方法既找回更多行，也显著降低方向误差。

走廊图像代理也得到改善：合计左右边界成对正确率0.9160、走廊中心MAE 0.0146、
受支持帧valid召回0.9248。但不安全false-valid率仍为0.1314，复用验证分区为0.2807，
没有达到合同≤0.05的安全门。这说明“多行几何提取”已经过关，“可靠走廊拒识”仍未过关。

模型纯GPU热图推理实测约3.75 ms/帧；该值不含Day61/62预处理和几何解码，不能替代
合同要求的640×360 CPU端到端计时，完整性能验收留到Day66。

## 23. 新增代码、测试与本地产物

关键新增接口：

- `CropRowLine`、`MultiRowPrediction`：多行与走廊结构；
- `extract_multirow_geometry`：传统多带基线；
- `match_ordered_crop_rows`：顺序保持的一对一行匹配；
- `derive_image_corridor`：左右相邻边界与中线；
- `TinyRowUNet`、`ResNet18RowUNet`：二维中心线网络；
- `prepare_centerline_tensor`、`decode_centerline_heatmap`：四通道输入与多行解码；
- `split_crossfit_folds`：确定性标签排除折；
- `finalize_day63_resnet_oof_from_cache`：OOF结果复核与交付产物。

最终测试为：

```text
34 passed
```

本地主要产物：

```text
day63_results_resnet18_oof.json
geometry_metrics_resnet18_oof.csv
geometry_comparison_resnet18_oof.jpg
day63_resnet18_centerline_model.pt
day63_results_centerline_round1_rejected.json
```

完整重跑命令如下；默认会复用已校验的OOF概率缓存，增加
`--no-reuse-oof-cache` 才会重新训练全部折模型：

```powershell
D:\conda\envs\forest-species\python.exe -X utf8 `
  63_crop_row_geometry_extraction\code\day63_crop_row_geometry.py `
  --train-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\train_data `
  --train-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_train_development_manifest.jsonl `
  --validation-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\validation_data `
  --validation-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_validation_development_manifest.jsonl `
  --output-dir D:\DL_code\data\crop_row_perception\day63_crop_row_geometry `
  --study resnet_oof `
  --pretrained-weights C:\Users\13262\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
```

CRDLD许可尚未确认，因此图片、逐图概率、模型和运行结果继续只保存在
`D:/DL_code/data/crop_row_perception/day63_crop_row_geometry/`，不上传GitHub；Git只同步代码、
测试、笔记和不含原始数据的项目进度说明。

## 24. Day63重学最终评价

作为Day63“检测所有可评价可见作物行并形成图像走廊候选”的学习成果，本版达到优秀：

- 两个开发分区的标签排除OOF多行P/R、位置和方向门全部通过；
- 输出不再固定为一条中央行；
- 左右边界、走廊中线和中央作物行降级规则已落地；
- 旧单行方法和所有主要失败轮次均保留；
- 没有访问CRDLD test_data或RowDetr冻结外部集。

但作为真实农业机器人完整视觉仍不能称优秀，当前明确阻断项是：

- 不安全false-valid率未达到0.05；
- 没有真实走廊、机器人车体、地头和负样本真值；
- 没有相机标定、米制测量、视频时序、转弯重选和外部冻结测试；
- 验证开发数据已经参与方法选择，最高泄漏分组仍未知。

因此正确状态是：

```text
DAY63_MULTIROW_RELEARNING_COMPLETE
TRAIN_DEVELOPMENT_OOF_MULTIROW_GEOMETRY_PASS
REUSED_VALIDATION_DEVELOPMENT_OOF_MULTIROW_GEOMETRY_PASS
CORRIDOR_BOUNDARY_AND_CENTER_PROXY_PASS
UNSAFE_FALSE_VALID_GATE_FAIL
CRDLD_INTERNAL_TEST_NOT_ACCESSED
ROWDETR_FROZEN_EXTERNAL_NOT_ACCESSED
READY_FOR_DAY65_TEMPORAL_AND_REJECTION_WORK
REAL_ROBOT_RELIABILITY_NOT_ESTABLISHED
```

Day64 可以使用今天输出的 `near_x_norm`、`far_x_norm` 和 `heading_deg` 学习图像中心偏移、消失点与相机坐标测量，但不能把图像归一化偏移解释为米制误差。

## 附录A：为什么曾经需要单行第二版

第一版虽然通过了“均值误差 + 有效率”门槛，但还不能令人满意：在248张已复用的
`validation-development` 上，位置P90为0.0615、方向P90为9.615°，并且只有
60.89%的全部帧同时满足位置误差不超过0.05和方向误差不超过5°。对比图还显示，
单条支撑线搜索可能被相邻作物行吸引。因此第一版保留为对照基线，不再作为Day64首选输入。

## 附录B：单行第二版方法——多尺度掩膜特征 + Extra Trees端点回归

第二版不修改Day61颜色阈值，也不修改Day62冻结形态学。它只改变几何层：

1. 将Day62二值掩膜缩放为20×12，得到240维粗布局特征；
2. 在16个水平带分别计算前景密度、横向均值和横向标准差，得到48维透视布局特征；
3. 将整图列占用率压缩为32维，补充全局作物行排列信息；
4. 合计320维固定长度特征，同时回归 `near_x_norm` 和 `far_x_norm`；
5. 用240棵Extra Trees组成集成，以各树端点预测的离散程度记录不确定性；
6. 继续沿预测线测量冻结掩膜支撑度，输出 `valid/degraded/reject`，置信度仍不是概率。

选择规则在运行验证集前冻结：1250张训练开发图确定性分成5折；第二版必须在每一折
保持有效率至少0.85、位置MAE不得比第一版差0.005以上、方向MAE至少改善0.5°、
方向P90不超过5°，并且双阈值达标率至少逐折提升0.10。

## 附录C：单行第二版五折结果

四组Extra Trees候选全部通过预注册的逐折改进门。按“五折平均双阈值达标率最高，
再比较位置和方向误差”的规则，训练阶段选中：

```text
extra_depth12_leaf2
n_estimators = 240
max_depth = 12
min_samples_leaf = 2
max_features = 0.7
```

选中方案的五折平均指标：

| 指标 | 第一版五折约值 | 第二版五折平均 |
| --- | ---: | ---: |
| valid fraction | 0.8632 | 0.9264 |
| 近端位置MAE | 0.0280 | 0.0205 |
| 方向MAE | 3.532° | 1.944° |
| 方向P90 | 约7.52° | 3.800° |
| 全帧双阈值达标率 | 约68.7% | 88.24% |
| 全链路中位耗时 | 约36.45 ms | 34.30 ms |

这里的五折只证明在CRDLD训练开发范围内稳定改善，不是外部泛化证据。

## 附录D：已复用开发集上的单行第一版与第二版对照

这248张图在Day62和Day63第一版已经参与开发，所以本节只能叫“开发对照”，不能叫
独立确认。第二版所有预注册开发门均通过：

| 指标 | 第一版 | 第二版 | 结论 |
| --- | ---: | ---: | --- |
| valid fraction | 0.8589 | 0.9476 | 提高8.87个百分点 |
| 近端位置MAE | 0.0320 | 0.0239 | 降低25.3% |
| 方向MAE | 4.260° | 1.809° | 降低57.5% |
| 位置P90 | 0.0615 | 0.0545 | 改善但仍有长尾 |
| 方向P90 | 9.615° | 3.994° | 已低于5°门槛 |
| 双阈值达标率 | 60.89% | 82.66% | 提高21.77个百分点 |
| 最大位置误差 | 0.3610 | 0.1384 | 明显下降但未消失 |
| 最大方向误差 | 21.983° | 6.599° | 明显下降但未消失 |
| 全链路中位耗时 | 38.21 ms | 36.60 ms | 未以精度换明显延迟 |

对比图中多数第二版预测更贴近中央参考行，但样本37等仍存在明显位置错行。这一负面结果
被保留，后续应在Day67归入失败案例，在Day68只能选择一个主要原因做受控改进。

## 附录E：单行第二版产物与复现

本地产物：

```text
D:/DL_code/data/crop_row_perception/day63_crop_row_geometry/day63_results_v2.json
D:/DL_code/data/crop_row_perception/day63_crop_row_geometry/geometry_metrics_v2.csv
D:/DL_code/data/crop_row_perception/day63_crop_row_geometry/geometry_contact_sheet_v2.jpg
D:/DL_code/data/crop_row_perception/day63_crop_row_geometry/day63_geometry_v2.joblib
```

复现时在原命令增加 `--study v2`；当前命令行默认也是第二版。模型文件保存了估计器、
320维特征合同、端点目标合同、选中配置和Day62冻结配置，不包含原始CRDLD图像。

## 附录F：单行第二版历史评价与Day64交接

作为Day63课程成果，第二版达到优秀并足以进入Day64：完整覆盖ROI/透视限制、Hough基线、
稳健几何回归、五折选择、尾部指标、不确定性、模型持久化、逐图CSV和可视化审计；22项
Day63测试通过，Day61/62保持冻结。

作为真实农业机器人算法仍不能称优秀：没有未触碰确认集、同源内部测试尚未访问、
RowDetr冻结外部测试尚未访问、缺少目标域拒识负样本、没有相机标定与米制误差，且个别
相邻行误选仍然存在。Day64可以使用第二版的近/远端点定义图像偏移和方向代理，但不得把
这些归一化量直接解释为物理距离或控制可靠性。

```text
DAY63_V2_LESSON_COMPLETE
TRAIN_FIVE_FOLD_V2_IMPROVEMENT_GATE_PASS
SELECTED_EXTRA_DEPTH12_LEAF2
REUSED_VALIDATION_DEVELOPMENT_ALL_GATES_PASS
NO_UNTOUCHED_CONFIRMATION_REMAINS
SAME_SOURCE_INTERNAL_BENCHMARK_NOT_ACCESSED
FROZEN_EXTERNAL_NOT_ACCESSED
REJECT_AWARE_EVALUATION_BLOCKED
READY_FOR_DAY64_DEFINITION_WORK
REAL_ROBOT_RELIABILITY_NOT_ESTABLISHED
```
