# Day64：相机坐标与测量——图像偏移、方向、消失点与标定边界

## 1. 今日目标

Day64不再优化Day63的作物行检测，而是回答一个更基础的问题：Day63输出的两个归一化
端点究竟能测量什么，哪些量在缺少相机标定时不能声称已经得到。

完整链路为：

```text
Day63冻结近点/远点
        ↓
图像平面归一化偏移和方向代理
        ↓
单线外推点与真实消失点条件分离
        ↓
相机内参存在时才能得到相机射线
        ↓
地面变换存在时才能讨论米制偏移和地面方向
```

## 2. 冻结输入

今天直接加载Day63保存的 `day63_geometry_v2.joblib`：

```text
selected model: extra_depth12_leaf2
feature length: 320
near evaluation y: 0.90
far evaluation y: 0.40
```

Day61颜色、Day62形态学和Day63 Extra Trees均未重新训练或调参。Day63的
`valid/degraded/reject`、置信度和不确定性原样向下游传播；Day64不能把降级或拒绝结果
提升为有效测量。

## 3. 三种不能混淆的坐标

### 3.1 图像归一化坐标

图像左上角为 `(0, 0)`，右下角为 `(1, 1)`。它与图像分辨率无关，适合比较不同尺寸
图像，但不是物理坐标。

### 3.2 像素坐标

对宽 `W`、高 `H` 的图像：

```text
u = x_norm × (W - 1)
v = y_norm × (H - 1)
```

使用 `W-1` 和 `H-1` 可以让归一化坐标1准确落在最后一个像素。

### 3.3 相机与地面坐标

相机射线需要真实内参矩阵 `K`；地面米制坐标还需要畸变、相机安装姿态和有效地面
变换。当前数据没有这些标定证据，所以正式结果必须阻塞这两层输出。

## 4. 图像中心偏移

Day64继续使用Day59预先确定的符号：

```text
lateral_offset_norm = near_x_norm - 0.5
```

- 等于0：近端作物行位于图像中心；
- 大于0：作物行位于图像右侧；
- 小于0：作物行位于图像左侧。

像素偏移为：

```text
lateral_offset_px = lateral_offset_norm × (W - 1)
```

它们都只是图像平面量，不能解释成“机器人偏了多少米”。

## 5. 方向角定义

归一化方向代理与Day63完全一致：

```text
heading_proxy_deg = degrees(
    atan2(far_x_norm - near_x_norm, near_y_norm - far_y_norm)
)
```

从近点指向远点，图像上方视为前进方向。远点在近点右侧时为正，左侧时为负。

另外计算 `pixel_slope_deg` 只用于可视化。它使用像素横纵距离，因此会受宽高比影响；
`heading_proxy_deg` 虽然分辨率无关，仍然不是相机标定后的真实航向角。

## 6. 单线外推点不等于消失点

一条中央线只能定义直线。将它外推到 `y=0.25` 得到的是
`horizon_intercept_norm`，只是单线外推代理。

真实消失点至少需要两条非平行图像线。Day64实现齐次直线表示，并用迭代重加权最小
二乘估计共同交点：

```text
l_i = p_i1 × p_i2
A [x, y]^T = -c
```

少于两条线、平行线或病态线组会返回不可用。当前Day63只输出一条中央行，因此真实数据
必须记录：

```text
BLOCKED_SINGLE_LINE
```

合成多线测试只能验证数学实现，不能把当前真实输出变成消失点证据。

## 7. 相机射线

如果存在经过验证的内参矩阵：

```text
ray = normalize(K^-1 [u, v, 1]^T)
```

它得到的是相机坐标系中的单位射线，而不是地面点。代码对合成内参完成了中心射线和
右侧射线测试；真实数据没有内参，输出为：

```text
BLOCKED_NO_CALIBRATION
```

## 8. 地面米制测量

只有图像到地面的有效单应变换，并且其输出坐标明确为 `X向右、Z向前、单位为米`，才可
计算：

```text
lateral_offset_m = X_near - X_robot_center
heading_deg = degrees(atan2(X_far - X_near, Z_far - Z_near))
```

代码使用单位矩阵合成夹具验证公式，但真实CRDLD没有相机安装高度、内外参或地面变换，
因此正式输出为：

```text
BLOCKED_NO_GROUND_TRANSFORM
```

## 9. 测试驱动过程

先创建15项测试并运行，得到预期RED：

```text
ModuleNotFoundError: No module named 'day64_camera_measurement'
```

实现后覆盖：

- 归一化角点到像素的映射与非法输入；
- 中心、左右偏移和垂直方向；
- 左右方向角符号；
- 不同分辨率下归一化测量一致；
- Day63降级和拒绝状态不被提升；
- 单线只输出外推代理；
- 少于两线和平行线拒绝；
- 合成多线恢复已知消失点；
- 缺少内参时阻塞相机射线；
- 合成内参反投影单位射线；
- 缺少地面变换时阻塞米制测量；
- 合成地面变换验证偏移和方向；
- 小型端到端研究写出JSON、CSV和对比图。

实现后的Day64测试：

```text
15 passed
```

## 10. 真实开发数据运行

使用248张已经参与Day62/63开发的 `validation-development` 图片。它们不是未触碰验证集，
今天不再选参，只检查冻结模型和坐标定义的一致性。

| 项目 | 结果 |
| --- | ---: |
| 图片数 | 248 |
| valid fraction | 0.9476 |
| degraded fraction | 0.0524 |
| reject fraction | 0.0000 |
| 有效帧平均绝对图像偏移 | 0.0629 |
| 有效帧平均绝对方向代理 | 2.826° |
| Day63方向复算最大差异 | 0.0° |

下面三项只是从同一批预测重新计算得到的Day63继承指标，不是Day64新增性能：

| Day63继承指标 | 数值 |
| --- | ---: |
| 近端位置MAE | 0.0239 |
| 方向MAE | 1.809° |
| 双阈值达标率 | 82.66% |

## 11. 可视化检查

8张图覆盖最佳、典型、最差位置、最差方向、最低和最高不确定性。每行显示：

```text
原图 | Day62冻结掩码 | 中心线/预测/真值 | 状态与测量边界
```

- 青色：图像中心；
- 黄色：Day63预测；
- 紫色：标签中央参考行。

样本37和132仍保留明显位置误差，说明坐标转换不能修复上游错行。它们应留到Day67失败
分类和Day68单原因改进，而不是在Day64重新调整Day63。

## 12. 代码结构

主文件：

```text
64_crop_row_camera_coordinates_measurement/code/day64_camera_measurement.py
```

关键函数：

- `normalized_to_pixel`：归一化坐标转像素；
- `image_coordinate_measurement`：偏移、方向和单线外推代理；
- `estimate_vanishing_point`：多线稳健交点与病态拒绝；
- `project_pixel_to_camera_ray`：有内参时反投影相机射线；
- `ground_plane_measurement`：有地面变换时输出米制合同；
- `run_day64_study`：加载冻结Day63模型并生成完整结果。

## 13. 本地产物

```text
D:/DL_code/data/crop_row_perception/day64_camera_measurement/day64_results.json
D:/DL_code/data/crop_row_perception/day64_camera_measurement/coordinate_metrics.csv
D:/DL_code/data/crop_row_perception/day64_camera_measurement/coordinate_contact_sheet.jpg
```

数据图片、逐图CSV和运行结果不进入Git。

## 14. 复现命令

```powershell
D:\conda\envs\forest-species\python.exe -X utf8 `
  64_crop_row_camera_coordinates_measurement\code\day64_camera_measurement.py `
  --day63-model D:\DL_code\data\crop_row_perception\day63_crop_row_geometry\day63_geometry_v2.joblib `
  --validation-root D:\DL_code\data\crop_row_perception\sources\crdld_v2_1\data\validation_data `
  --validation-manifest projects\04_crop_row_perception\data\scoped_crdld\crdld_validation_development_manifest.jsonl `
  --output-dir D:\DL_code\data\crop_row_perception\day64_camera_measurement `
  --count 8
```

## 15. Day64评价与Day65交接

作为Day64学习任务，本次结果达到要求：图像中心、归一化偏移、方向代理、消失点条件、
相机射线和地面测量的层次已经明确；冻结模型复算完全一致；无法支持的物理量被显式阻塞。

作为真实农业机器人测量系统不能称为完成：没有真实相机内外参、畸变标定、相机安装姿态、
地面变换、真实消失点、米制误差评价或机器人闭环验证。

Day65可以对 `lateral_offset_norm`、`heading_proxy_deg`、`confidence`、`uncertainty` 和
`status` 做视频时序稳定，但不得对不存在的米制量进行平滑。

```text
DAY64_LESSON_COMPLETE
DAY63_V2_INPUT_FROZEN
IMAGE_COORDINATE_CONTRACT_PASS
DAY63_HEADING_REPRODUCTION_EXACT
VANISHING_POINT_BLOCKED_SINGLE_LINE
CAMERA_RAY_BLOCKED_NO_CALIBRATION
METRIC_MEASUREMENT_BLOCKED_NO_GROUND_TRANSFORM
READY_FOR_DAY65_TEMPORAL_STABILITY
REAL_ROBOT_CONTROL_NOT_ESTABLISHED
```
