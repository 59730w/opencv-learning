# Day66：冻结多作物行管线的完整离线视频 Pilot

> 最终结论：Day66 第二版的课程与离线工程目标通过。冻结的 Day63 感知与 Day65 时序配置在
> 25 段开发视频、10,995 帧上生成了逐帧结构化输出和连续可视化，9 项结构门及
> `640×360` CPU 中位耗时门全部通过。该结果仍不是带真值的真实视频准确率、外部泛化、
> 实车安全或闭环控制证据。

## 1. 今天学习和完成了什么

Day65 已经证明时序层可以减少抖动和状态闪烁，但只保存了少量演示视频。Day66 不重新训练
模型，也不围绕演示画面调整阈值，而是把冻结管线整理成完整、可复现、可审计的离线 Pilot：

```text
开发 RGB 视频
  → 冻结 Day61/62 颜色与形态学特征
  → 冻结 Day63 RGB+mask ResNet18 多行检测
  → Day64 图像坐标走廊测量
  → 冻结 Day65 光流、身份关联、Kalman 与拒绝状态机
  → 逐帧合同 JSONL + 2×连续叠加视频 + 完整性/性能/视觉审计
```

Day66 重点学习的是模型之外的完整视觉系统工程：配置冻结、输出合同、视频编解码、状态语义、
可复现哈希、分层抽样、运行时间边界以及失败结果如何保留。

## 2. 数据范围与不可跨越的冻结边界

- 数据：LeCropFollow CROW；原视频均为机器人视角 `320×180`；
- 处理角色：`temporal_development`、`shifted_development`；
- 正式处理：25 段，10,995 帧；
- 未处理角色：2026-05-05 的 6 段 `frozen_same_source_holdout`；
- 冻结视频未打开、未推理、未渲染、未抽样，也未参与运行时间或显示选择；
- 数据许可仍为 `NOT_VERIFIED`，所以视频、逐帧输出和联系表只留在本地；
- RGB 帧与数据表时间戳仍不能一一对应，因此不声明毫秒级传感器同步或物理运动估计。

冻结同源视频留到 Day69，在盲标注、评价规则和参数完全锁定后只运行一次。同源留出即使届时
使用，也不是独立外部证据。

## 3. 冻结配置与双层状态合同

Day66 的 `day66_frozen_config.json` 与 Day65 正式结果逐字段相等：

| 参数 | 冻结值 |
|---|---:|
| `association_gate` | 0.25 |
| `max_missing_frames` | 8 |
| `confirm_frames` | 2 |
| `recovery_frames` | 1 |
| `switch_confirm_frames` | 6 |
| `confirmation_window_frames` | 8 |
| `process_variance` | `1e-5` |
| `measurement_variance` | `0.0025` |
| `minimum_row_confidence` | 0.12 |
| optical flow | enabled，最长补测 2 帧 |

Day65 类构造器还保留着早期默认值，因此 Day66 禁止隐式调用 `TemporalConfig()`。程序必须读取
冻结 JSON，再与 Day65 结果比对；任何一个值漂移都会立即失败。CLI 也不暴露上述调参选项。

输出分成两层，避免把等待确认的状态误当导航：

- `pilot_state`：`valid/candidate/degraded/reject`，用于解释时序内部过程；
- `status`：合同公开状态 `valid/degraded/reject`，其中 `candidate` 对外映射为 `degraded`；
- `navigation_available`：只有 `pilot_state=valid` 且左右边界完整时才为 `true`；
- 其他状态的左右边界、中心线、偏移、方向、消失点和行距全部为 `null`。

每帧同时保存所有跟踪作物行及 ID、左右边界、走廊中心两点、归一化偏移、图像方向代理、
多线消失点、归一化行距、置信度、状态、原因和光流诊断。偏移、方向和行距仍是图像量，
不是米、真实航向或机器人安全余量。

## 4. 连续叠加与可读性修正

第一次真实冒烟视频虽然能完整解码，但 `320×180` 上的状态与原因文字拥挤、截断。该问题只属
显示层，修正前先增加失败测试，然后：

- 叠加视频以 2× 尺寸 `640×360` 写出，感知仍读取原始帧；
- 字号、线宽和标签位置随显示宽度缩放；
- 原因文字按 OpenCV 实际测得的像素宽度自动换成最多两行；
- 普通跟踪行为白色，选中左/右边界分别为紫色/蓝色；
- 绿色中心和消失点只在 `valid` 出现；
- `candidate/degraded/reject` 保留行 ID 和失败原因，但不画导航中心。

冒烟片段48帧的叠加视频与JSONL全部对齐，修正后再次生成联系表确认文字可读。

用户复核第一版显示合同时发现，“走廊中心线”和“方向代理”虽然在结构化字段中分别存在，
画面却只用绿色中心线间接表达方向，不能算两个项目都被明确展示。因此 Day66 第二版仅升级
显示层，不修改模型、权重、检测阈值、时序阈值或状态结果：

- 每个 `valid` 帧给选中边界增加明确的 `LEFT=ID...`、`RIGHT=ID...` 标签；
- 绿色线继续表示走廊中心，右上角独立黄色 `DIR` 箭头表示图像方向代理；
- HUD 同时显示 `heading=+/-x.xdeg`；非 `valid` 显示 `heading=n/a`；
- 所有文字增加黑色描边，避免白色文字在天空、强光和浅色叶片上消失；
- 左右标签移动到边界中段，与方向 HUD 分区，修复真实预览中的标签重叠；
- 新增6个显示合同测试，逐项覆盖有效方向代理、左右标签、非有效禁用、布局与亮背景对比度。

第二版完整重跑后，四态计数与第一版完全相同，说明升级只改变展示像素，没有改变感知或
状态决策。新的6类联系表均从第二版连续视频重新生成，未复用第一版截图。

## 5. 第一轮性能失败及诊断

第一轮完整 Pilot 的9项结构检查通过，但预注册CPU门失败：

| 项目 | 第一轮结果 |
|---|---:|
| CPU输入与批量 | `640×360`、batch 1、4线程 |
| 中位耗时 | 77.37 ms/帧 |
| P95耗时 | 87.64 ms/帧 |
| 目标 | 中位数 ≤ 50 ms |
| 结论 | `FAIL` |

失败结果保存在本地 `attempt_01_runtime_failed.json`。随后仅在开发片段上做运行时间诊断；1、2、
4、8线程中位数约为108.45、83.88、73.41、73.74 ms，说明线程数不是根因。

分项分析显示冻结特征构建和 PyTorch eager 模型推理占主要成本。优化没有改变模型权重、训练、
检测阈值或时序阈值：

1. CPU模型采用 `channels_last + torch.jit.optimize_for_inference`；
2. 48个开发帧与 eager 结果的解码行数量完全相同，端点/置信度最大绝对差为
   `4.0233e-6`，时序状态、导航可用性和边界ID在 `1e-5` 容差内一致；
3. Gray-World通道均值改为 OpenCV 快速路径，并以随机图像逐像素对照冻结 NumPy 参考；
4. 二值掩码检查由多次 `np.unique` 全排序改为一次布尔扫描；
5. 透视连通域过滤由“每个组件重新扫描整张标签图”改为一次保留查找表映射，并与原循环输出
   逐像素相等；
6. 模型仍按合同评价 `640×360` 输入；光流分支不再把原本 `320×180` 的冻结像素阈值错误地
   放大到640，而是保持最大工作尺度 `320×180`。所有行坐标仍为归一化坐标。

这是一轮输出等价的执行层优化，不是为了提高有效帧比例而重调感知。

## 6. 最终完整 Pilot 结果

| 项目 | 最终结果 |
|---|---:|
| 视频数 | 25 |
| 总帧数 | 10,995 |
| `valid` | 490（4.4566%） |
| `candidate` | 47 |
| `degraded` | 10,436 |
| `reject` | 22 |
| 逐帧JSONL | 25个 |
| 连续叠加MP4 | 25个 |
| 分层联系表 | 6张 |
| 非valid导航字段泄漏 | 0 |
| CUDA完整感知均值 | 14.67 ms/帧 |
| 2×绘制/编码均值 | 7.18 ms/帧 |

9项结构门全部通过：开发角色非空、角色范围正确、原视频全部解码、叠加视频全部重新解码、
源帧/JSONL/叠加帧一一对应、冻结配置精确匹配、冻结角色结构排除、冻结帧未访问、导航字段
仅在valid出现。

最终CPU基准：

| 项目 | 结果 |
|---|---:|
| 输入 | `640×360` |
| 设备/线程 | CPU / 4线程 |
| batch | 1 |
| 预热/计时帧 | 10 / 48 |
| 中位耗时 | **41.14 ms/帧** |
| P95 | 49.33 ms/帧 |
| 最小/最大 | 30.70 / 53.35 ms/帧 |
| 中位≤50 ms | `PASS` |

计时包括单帧冻结模型推理、行合理性过滤、光流补测和时序跟踪；视频解码、缩放和叠加编码
在计时边界外。多次第一版优化复测的中位数为43.31～44.43 ms；第二版最终复测为41.14 ms，
均通过预注册中位门。第二版 P95 为49.33 ms，最大值仍为53.35 ms，尾延迟没有被隐藏。结果
只代表当前笔记本，不证明真实机器人持续实时性。绘制/编码耗时增加来自方向 HUD、边界标签和
文字描边，且按预注册边界不计入感知运行时间门。

## 7. 分层视觉审计与负证据

从完整10,995帧中按episode和帧序确定性等距抽取：`valid/candidate/degraded/reject`各6帧，
状态切换6帧，光流补测6帧。另对一段temporal-development和一段shifted-development视频
各抽取6个全时段位置，并对25个叠加视频做完整解码检查。

观察结果：

- 颜色和状态语义一致；非valid画面没有绿色中心；
- `candidate`清楚表示等待连续确认；
- 缺边、中央作物行、换走廊等待和无活跃轨迹均给出可解释原因；
- 多数valid抽样的中心位于所选两条边界之间；
- 高遮挡和上游仅检测一行时大量输出degraded，符合保守策略；
- 发现一个高风险候选：`crow_2026-01-22-08-38-52_ep0`第893帧被标为valid，但画面近似
  草地/围栏，叠加线视觉上不像可靠作物行。这是疑似域外false-valid，应成为Day67优先失败组。

最后一点不能直接记为统计错误，因为视频没有逐帧走廊真值；也不能在Day66针对该抽样继续调参。
它证明了为什么结构门通过不等于真实视频安全通过。

## 8. 测试驱动过程

新增测试均先看到预期失败，再写最小实现。覆盖：冻结配置漂移、双层状态合同、非valid字段清空、
有效边界投影、叠加颜色与消失点、文字换行、视频/JSONL帧对齐、状态跨episode重置、冻结角色
排除、输出视频重解码、结构门、CPU基准边界、CLI不暴露调参、TorchScript/eager等价、多尺度
光流以及分层审计。

Day61增加Gray-World像素等价测试；Day62增加快速二值检查及透视连通域参考循环逐像素等价测试。
最终联合运行Day61–66共138项相关测试，全部通过；另对10,995条JSONL合同和25个覆盖视频做了
独立逐字段、逐帧核验，非valid导航泄漏与非valid原因缺失均为0。

## 9. 复现命令

```powershell
& 'D:\conda\envs\forest-species\python.exe' -X utf8 `
  'D:\opencv-learning\66_crop_row_offline_video_pilot\code\day66_offline_video_pilot.py' `
  --manifest 'D:\DL_code\data\crop_row_perception\sources\lecrop_data\crow_video_manifest.json' `
  --checkpoint 'D:\DL_code\data\crop_row_perception\day63_crop_row_geometry\day63_resnet18_centerline_model.pt' `
  --day65-result 'D:\DL_code\data\crop_row_perception\day65_video_temporal_verified\day65_results.json' `
  --output-dir 'D:\DL_code\data\crop_row_perception\day66_offline_video_pilot' `
  --device cuda `
  --cpu-benchmark-video 'D:\DL_code\data\crop_row_perception\sources\lecrop_data\rgb_episodes\crowfollow_lecropfollow_exp_2026-01-26-12-30-17_ep0.mp4' `
  --cpu-benchmark-frames 48 `
  --cpu-benchmark-warmup 10 `
  --cpu-threads 4
```

冻结配置默认来自与脚本同目录的 `day66_frozen_config.json`。正式结果标记为
`DAY66_OFFLINE_PILOT_COMPLETE`，工程门为 `true`。

## 10. 本地成果与证据边界

```text
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\day66_results.json
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\day66_visual_audit.json
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\*_day66_overlay.mp4
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\*.jsonl
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\day66_audit_*.jpg
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\attempt_01_runtime_failed.json
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\cpu_thread_diagnostic.json
D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\cpu_backend_optimization.json
```

允许的结论：冻结管线已形成结构完整、输出可解释、可复现且当前机器CPU中位耗时达标的开发视频
离线Pilot。

禁止的结论：真实视频安全率、独立外部泛化、米制走廊、真实航向、实时实车控制或农业机器人
部署已经通过。`real_video_safety_gate`继续是
`BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH`。

## 11. Day67交接

Day67不应立即再次调模型。先用盲于冻结组的开发证据建立失败分组，至少包括：疑似域外
高风险valid、中央行占据、左右缺边、严重遮挡、强光/模糊、走廊身份切换和长期上游无检测。
只有完成可复核标签或一致的人工审计协议后，Day68才允许针对一个预注册的主导原因做一次受控改进。

用户于2026-09-07确认第二版达到要求，并授权同步项目README、更新学习Skill和上传GitHub。
