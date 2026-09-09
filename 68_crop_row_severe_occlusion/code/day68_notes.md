# Day68：只针对严重遮挡做一次受控改进

## 1. 今天真正解决的问题

Day67没有授权继续普遍提高 `valid` 比例，而是把唯一可行动目标冻结为
`severe_occlusion`：在手部、近景叶片等严重遮挡之后，Day66偶尔会过早恢复
`valid`，从而发布并不可靠的走廊导航。本日只增加一个后投影安全守卫，不改
Day61～66的分割、几何、跟踪、状态机或走廊选择参数。

目标合同仍然是：

- 检测/跟踪行和稳定ID可以继续作为诊断信息保存；
- 只有 `valid` 可以发布左右边界、中心线、偏移、方向、消失点与行距；
- `candidate/degraded/reject` 的所有导航字段必须为空；
- 冻结测试视频在Day69以前不得打开、解码、推理或渲染。

Day68的证据只能回答“这项开发期改动是否拦住Day67已复核的严重遮挡疑似不安全
valid，并付出多少覆盖率代价”，不能回答真实视频准确率或机器人安全率。

## 2. 输入与冻结边界

- 输入：Day66的25段开发视频逐帧JSONL，共10,995帧；
- 角色：仅 `temporal_development` 和 `shifted_development`；
- 复核依据：Day67第三版536个事件标注，其中全部296个valid事件均被复核；
- 目标：3个 `severe_occlusion + suspected_unsafe + valid` 事件，恰好3帧；
- 冻结输入哈希写入 `day68_frozen_config.json`，运行时不一致即失败；
- 六段 `frozen_same_source_holdout` 视频访问帧数为0。

这3个目标帧都位于 `crow_2026-01-26-10-49-17_ep0` 的62、94、143帧。它们分别
出现在近景茎叶、手部遮挡和大片近景叶片情况下，且在恢复valid前已经连续8、31、48帧
没有可用导航。

## 3. 探索过但没有采用的方法

### 3.1 冻结ResNet18外观嵌入

使用冻结ImageNet ResNet18特征和按episode留出的逻辑回归，事件级AP约为0.14～0.20。
它无法在捕获全部3个目标的同时保留至少90%的Day66 valid帧。只有3个正目标时训练一个
新视觉分类器也极易记住单一episode，因此不采用。

### 3.2 手工外观特征分类器

颜色、植被、肤色、边缘、边界支撑、置信度和恢复长度组成的逻辑回归，最佳未加权事件级
AP约0.291。它同样不能满足“3/3目标全捕获 + valid至少保留90%”。这个负结果说明，
Day68不应伪装成已学会通用遮挡分类器。

## 4. 第一版及其失败原因

第一版规则是：

```text
前序连续非valid帧数 >= 8
AND 左右边界最小support_band_count <= 6
```

该规则确实拦住3/3目标，导航泄漏为0，valid保留率为451/490=92.04%，达到预设90%
下限。但逐一回查39个触发帧后，36帧被Day67标为 `no_failure_visible + appropriate`。
因此第一版虽然“过门”，却靠过度拒绝换取结果，不够令人满意，保留为失败轮次而不作为最终方案。

## 5. 第二版最终方法

### 5.1 因果恢复候选

第二版仍先使用第一版的两个因果条件找“风险恢复候选”。只有候选帧才读取当前帧图像并
计算走廊可观测性，不使用未来帧：

```text
recovery_candidate = (preceding_nonvalid >= 8) AND (minimum_support <= 6)
```

### 5.2 走廊可观测性

用Day66提议的左右边界在图像高度 `y=0.40～0.90` 形成四边形。灰度图执行固定
`Canny(60, 140)`，走廊边缘占比定义为：

```text
edge_fraction = 走廊掩膜内Canny边缘像素数 / 走廊掩膜像素数
```

同时检查左右边界的near/far归一化横坐标是否超出 `[0, 1]`。最终触发条件为：

```text
guard = recovery_candidate
        AND (boundary_out_of_frame OR edge_fraction <= 0.22)
```

阈值敏感性结果为：

| edge上限 | 触发valid帧 | valid保留率 | 捕获目标 |
|---:|---:|---:|---:|
| 0.20 | 13 | 97.35% | 3/3 |
| **0.22** | **17** | **96.53%** | **3/3** |
| 0.24 | 23 | 95.31% | 3/3 |

选择0.22，是为了相对两个低边缘目标0.189和0.135保留少量余量；第62帧边缘占比虽为
0.251，但右边界越出图像，因此由明确的几何越界条件拦截。

### 5.3 触发后的状态

守卫只作用一帧。触发时把 `valid` 改为 `degraded`，清空全部导航几何，但保留
`crop_rows`、稳定ID及原始Day66诊断。下一帧重新从Day66状态独立判断，不永久锁死。

## 6. 全量结果

第二版在25段开发视频、10,995帧上的正式结果：

| 指标 | Day66基线 | Day68第二版 |
|---|---:|---:|
| valid | 490 | 473 |
| candidate | 47 | 47 |
| degraded | 10,436 | 10,453 |
| reject | 22 | 22 |
| 已复核严重遮挡疑似不安全valid事件 | 3 | 0 |
| 非valid导航泄漏 | 0 | 0 |

- valid保留率：`473 / 490 = 96.53%`；
- 守卫触发：17帧，分布于11个episode；
- 触发帧复核构成：3帧 `severe_occlusion + suspected_unsafe`，14帧
  `no_failure_visible + appropriate`；
- 相比第一版，正常复核帧的额外拒绝从36帧降至14帧；
- 25个JSONL和25个叠加MP4均生成；所有MP4再次完整解码，合计10,995帧且逐段帧数一致；
- 12项Day68测试通过，Day61～68联合175项测试通过。

工程门为 `PASS`。这个结论仅表示第二版明显优于第一版，并满足本日冻结的开发期目标；
14帧正常恢复仍是可见覆盖率成本，而不是被隐藏的成功。

## 7. 最终画面如何阅读

Day68继续复用Day66第二版叠加：所有活跃/跟踪作物行及稳定ID始终可见；只有valid状态
绘制LEFT/RIGHT边界、绿色走廊中心和黄色DIR方向代理；顶部显示帧号、状态、导航状态、
置信度与行数；底部显示降级/拒绝原因。新增第三行显示：

```text
D68 risk / guard / prior / support / edge / out
```

在守卫触发帧，行与ID仍保留，`pilot=degraded`、`nav=degraded`，不会画出可用走廊中心。

## 8. 复现命令

```powershell
& 'D:\conda\envs\forest-species\python.exe' -X utf8 `
  'D:\opencv-learning\68_crop_row_severe_occlusion\code\day68_severe_occlusion.py' `
  --day66-output 'D:\DL_code\data\crop_row_perception\day66_offline_video_pilot' `
  --manifest 'D:\DL_code\data\crop_row_perception\sources\lecrop_data\crow_video_manifest.json' `
  --day67-annotations 'D:\DL_code\data\crop_row_perception\day67_failure_taxonomy\day67_annotations_v3.jsonl' `
  --day67-results 'D:\DL_code\data\crop_row_perception\day67_failure_taxonomy\day67_results_v3.json' `
  --config 'D:\opencv-learning\68_crop_row_severe_occlusion\code\day68_frozen_config.json' `
  --output-dir 'D:\DL_code\data\crop_row_perception\day68_severe_occlusion' `
  --render-overlays
```

## 9. 文件与证据边界

- `day68_severe_occlusion.py`：因果恢复候选、走廊可观测性、导航清空、全量运行与可视化；
- `day68_frozen_config.json`：阈值、第一版失败、敏感性、输入哈希和证据边界；
- `test_day68_severe_occlusion.py`：状态合同、可观测性、越界、单帧互锁、输入隔离和视频对齐测试；
- 本地 `day68_results.json`、25个JSONL和25个MP4：正式开发期结果，不上传；
- 本地模型探索、联系图和临时计划位于忽略目录，不属于仓库交付。

真实视频缺少独立人工逐帧走廊有效性真值，Day67视觉决定也是模型辅助开发审查；因此
`real_video_safety_gate` 仍为
`BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH`。Day69只能在代码、配置和协议冻结后
一次性访问冻结集，并必须把同源内部测试、RowDetr外部正样本和缺失的目标域负样本分开报告。
