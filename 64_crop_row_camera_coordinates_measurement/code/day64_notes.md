# Day64：多作物行相机坐标与走廊测量（重学最终版）

> 当前结论：早晨的单中央作物行版本只保留为历史基线，不再作为项目四的当前输出。
> 本版直接接入Day63重学后的多行OOF结果，用相邻两条作物行确定机器人行间走廊；没有
> 相机标定时，所有量仍严格停留在图像坐标系。

## 1. 为什么Day64必须重学

原版把“最接近图像中央的一条作物行”当作参考，因此只能计算作物行相对图像中心的偏移，
不能回答机器人究竟应该走在哪两条作物行之间。Day63已经重学为多行检测，Day64若仍使用
单行，接口就会断裂，并可能把中央作物行本身误当成行驶位置。

本版链路改为：

```text
Day63逐帧OOF多行概率图
        ↓
有序作物行 + 多线稳健消失点
        ↓
图像中心左右最近的两条受支持作物行
        ↓
两条边界的中线 = 图像行间走廊中心
        ↓
归一化偏移、像素偏移、图像航向、图像行间距
        ↓
无标定时阻塞相机射线和米制测量
```

## 2. 输入和证据边界

- 输入：Day63最终ResNet18中心线模型产生的三折标签排除OOF概率图；
- 训练开发：1,250帧；复用验证开发：248帧；合计1,498帧；
- 解码参数沿用Day63冻结值：`peak_height=0.20`、`peak_prominence=0.03`、
  `peak_distance_norm=0.06`；
- Day61颜色、Day62形态学、Day63概率图均未在Day64重调；
- CRDLD `test_data`、RowDetr冻结外部集和目标域负样本均未访问。

所以结果是“同源、已参与方法选择的开发证据”，不是未触碰验证、外部泛化或实车安全证据。

## 3. 坐标合同

图像左上角为`(0,0)`，右下角为`(1,1)`。图像中心是`x=0.5`。归一化坐标转像素：

```text
u = x_norm × (W - 1)
v = y_norm × (H - 1)
```

所有行在`y=0.80`处排序。图像中心左侧最近的受支持行是左边界，右侧最近的受支持行是
右边界。两条边界的近点和远点分别取平均，构成走廊中心线。

```text
corridor_near_x = (left_near_x + right_near_x) / 2
corridor_far_x  = (left_far_x  + right_far_x)  / 2
lateral_offset_norm = corridor_near_x - 0.5
lateral_offset_px   = lateral_offset_norm × (W - 1)
heading_proxy_deg   = atan2(corridor_far_x - corridor_near_x, 0.90 - 0.40)
```

- 偏移为正：目标走廊中心在图像中心右侧；
- 偏移为负：目标走廊中心在图像中心左侧；
- 航向仍是图像空间代理角，不是标定后的车体航向角。

## 4. 防止把作物行当成道路

若`y=0.80`处存在满足`|row_x - 0.5| <= 0.04`的中央作物行，系统输出`degraded`，
不生成走廊中心。若只检测到图像中心一侧的作物行、边界支持不足、上游拒绝，或无法形成
稳定多线交点，也不把结果提升成`valid`。

这条规则解决的是语义方向错误：黄色/紫色作物行是边界证据，真正应走的位置是两条边界
中间的白色走廊中心线。

## 5. 多线真实消失点（图像空间）

每条作物行由`y=0.90`和`y=0.40`的两个端点定义。两条以上非平行行使用齐次直线方程，
并以迭代重加权最小二乘（IRLS）估计共同交点：

```text
l_i = p_i1 × p_i2
A [x_vp, y_vp]^T = -c
```

本版的消失点确实由多条检测行共同求得，不再是单线外推点。它仍然只是归一化图像坐标；
没有镜头畸变和相机内外参时，不能把它直接解释成地面上的真实方向或距离。

## 6. 行间距

将相邻有序行在`y=0.80`处的横向间隔记录为`row_spacing_norm`。本版报告预测与标签的
中位行间距差，但不输出米。真实米制行距需要相机标定、地面单应或深度信息。

## 7. 测试驱动过程

重学前先增加四项合同测试：

1. 四条作物行能选择中央相邻边界并产生走廊；
2. 中央作物行存在时必须降级，不能当成可行驶中心；
3. 缺少一侧边界时必须降级；
4. 能输出图像行间距，但米制测量必须阻塞。

首次运行得到4项失败、15项旧测试通过，失败原因是多行测量入口尚不存在。实现后再增加
小型多行端到端测试，最终结果：

```text
20 passed
```

## 8. 真实开发数据结果

| 指标 | 训练开发OOF | 复用验证开发OOF | 合计 |
|---|---:|---:|---:|
| 帧数 | 1,250 | 248 | 1,498 |
| 标签支持走廊帧数 | 607 | 191 | 798 |
| 支持走廊有效召回 | 90.94% | 97.38% | 92.48% |
| 不安全false-valid率 | 11.82% | 28.07% | 13.14% |
| 左右边界成对正确率 | 90.44% | 96.34% | 91.85% |
| 走廊中心MAE | 0.0144 | 0.0153 | 0.0146 |
| 走廊航向MAE | 1.203° | 1.138° | 1.186° |
| 消失点可用率 | 99.84% | 100.00% | 99.87% |
| 消失点中位误差 | 0.0186 | 0.0263 | 0.0195 |
| 中位行间距MAE | 0.0215 | 0.0630 | 0.0283 |

Day64测量门：

- 边界成对正确率不少于0.80：通过；
- 走廊中心MAE不大于0.05：通过；
- 走廊航向MAE不大于5°：通过；
- 消失点可用率不少于0.90：通过；
- 无标定时相机射线和米制测量全部阻塞：通过。

因此`day64_measurement_gate_passed = true`，本版作为Day64学习成果可以让人满意。

## 9. 没有通过的安全门

Day65交接门要求：支持走廊召回不少于0.80，且不安全false-valid率不大于0.05。前者通过，
后者13.14%未通过，因此`day65_safety_handoff_gate_passed = false`。

重学过程中实测并否决两种简单优化：

- 走廊宽度/典型行距比阈值：不能在保留足够真走廊的同时把false-valid降至5%；
- 更低热图阈值的一致性检查：对复用验证开发的false-valid几乎没有实质改善。

这里不继续反复调Day64坐标公式。该失败要求Day65利用视频时序、行身份连续性、置信度和
失效状态解决；目标域负样本仍是最终安全拒识的必要缺口。

## 10. 物理测量边界

当前缺少：

- 相机内参和镜头畸变系数；
- 相机相对车体/地面的外参或标定单应；
- 机器人真实宽度、轮迹和可通行性标签。

所以正式结果保持：

```text
camera ray: BLOCKED_NO_CALIBRATION
metric measurement: BLOCKED_NO_GROUND_TRANSFORM
real robot control: NOT ESTABLISHED
```

## 11. 代码与产物

主代码：`64_crop_row_camera_coordinates_measurement/code/day64_camera_measurement.py`

关键接口：

- `multirow_coordinate_measurement`：多行排序、左右边界、走廊中心、偏移和行间距；
- `estimate_vanishing_point`：多线IRLS消失点；
- `run_day64_multirow_study`：读取Day63 OOF缓存，生成逐帧CSV、汇总JSON和可视化；
- `image_coordinate_measurement`、`run_day64_study`：保留的单行历史基线。

本地产物：

```text
D:/DL_code/data/crop_row_perception/day64_camera_measurement/day64_results_multirow.json
D:/DL_code/data/crop_row_perception/day64_camera_measurement/coordinate_metrics_multirow.csv
D:/DL_code/data/crop_row_perception/day64_camera_measurement/coordinate_contact_sheet_multirow.jpg
```

旧的`day64_results.json`、`coordinate_metrics.csv`和`coordinate_contact_sheet.jpg`只作为
早晨单行版本历史留存，不代表当前结论。

## 12. 复现命令

```powershell
D:\conda\envs\forest-species\python.exe -X utf8 `
  64_crop_row_camera_coordinates_measurement\code\day64_camera_measurement.py `
  --mode multirow `
  --train-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\train_data `
  --train-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_train_development_manifest.jsonl `
  --train-probability-cache D:\DL_code\data\crop_row_perception\day63_crop_row_geometry\day63_resnet18_train_oof_probabilities.npz `
  --validation-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\validation_data `
  --validation-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_validation_development_manifest.jsonl `
  --validation-probability-cache D:\DL_code\data\crop_row_perception\day63_crop_row_geometry\day63_resnet18_validation_oof_probabilities.npz `
  --output-dir D:\DL_code\data\crop_row_perception\day64_camera_measurement `
  --count 8
```

## 13. Day64最终评价与Day65交接

Day64现在已经与新项目四对齐：检测到的多条作物行用于确定左右边界；白色中线才是图像
走廊中心；多条行共同估计图像消失点；中央作物行不会被直接当成行驶目标。核心测量门
全部通过，结果满意，无需再做第三轮Day64坐标优化。

Day65必须处理视频时序稳定、行身份连续、漏检恢复、走廊切换和false-valid抑制。即使
Day65完成，在Day69冻结外部测试、标定与实车数据到位之前，也只能称为离线农业机器人
视觉Pilot，不能称为可安全控制真实机器人。
