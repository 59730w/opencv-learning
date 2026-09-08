# Day67：把逐帧状态变成可复核失败事件，并为Day68只选一个原因

## 1. 今日目标与最终结论

Day67不修改Day61～66的模型、阈值、跟踪器或状态机。目标是把Day66的25段开发视频、
10,995帧输出转成可复核事件，分开记录“系统为什么降级”和“画面中可能发生了什么”，然后用
预先固定的门禁只为Day68选择一个可行动原因。

最终结果标记为 `DAY67_FAILURE_TAXONOMY_COMPLETE`：

- 25段开发视频/10,995帧完整对齐，形成4,809个连续事件；
- 第三版固定随机种子6703，保留全部296个valid事件，并按role/state/system-trigger的12层进行层内等概率无放回抽样，抽取
  240个non-valid事件，共复核536个事件；每个样本记录层总体、样本量、纳入概率和设计权重，
  高风险事件再检查完整上下文；
- 两轮检查全部34张原图/叠加联系表及15张non-valid专用联系表；对41个高风险事件另看上下文短片；
- 找到9个疑似不安全valid事件、13帧、2个episode；其中6个事件/10帧是在草地和围栏画面
  上继续输出valid，另3个事件/3帧发生在手部或近景叶片严重遮挡时；
- 加权Day68选择门为 `PASS`，唯一目标仍是 `severe_occlusion`；
- 真实视频安全门仍为 `BLOCKED_NO_FRAMEWISE_CORRIDOR_VALIDITY_GROUND_TRUTH`。

这里的“疑似不安全”是视觉复核发现的问题线索，不是真值判定，不能据此计算false-valid率。

## 2. 为什么用事件而不是把10,995帧当样本

同一视频中连续帧高度相关。把4,753个“双边界不足”帧当成4,753个独立失败会产生伪重复，
也会让长视频和长失败段支配结论。Day67的实验单位是一个episode中的连续事件：

```text
episode + pilot_state + normalized system trigger
```

valid/candidate以及重识别事件还纳入左右track pair，使身份变化不会被隐藏；普通上游缺失和
switch guard不因瞬时track churn被拆碎。每个事件保存起止帧、帧数、代表帧和前后12帧上下文。
结论以事件数和独立episode数为主，帧数只表示持续时间/影响。

## 3. 两层分类合同

### 3.1 自动系统触发器

系统触发器只规范化Day66已有reason，不推断视觉原因：

| system trigger | 事件 | 帧 | episode |
|---|---:|---:|---:|
| `insufficient_adjacent_boundaries` | 1,872 | 4,753 | 25 |
| `corridor_switch_guard` | 1,037 | 2,089 | 24 |
| `central_row_guard` | 781 | 2,319 | 25 |
| `upstream_row_rejection` | 770 | 1,275 | 24 |
| `initial_confirmation_guard` | 44 | 47 | 25 |
| `no_live_tracks` | 9 | 22 | 7 |

另有296个valid事件/490帧。四状态总数与Day66完全一致：candidate 47、degraded 10,436、
reject 22、valid 490。

### 3.2 视觉复核标签

视觉层允许严重遮挡、中央作物行、缺左右边界、光照/眩光、模糊、上游持续漏检、错误走廊/
身份、疑似OOD valid、状态机确认、未见明显失败和不可判定。它不会覆盖system trigger；同一事件
同时保留两个字段。每个标注还记录reviewability、secondary labels、导航判断和复核备注。

## 4. 确定性复核设计

复核清单包含全部296个valid事件，因此任何valid都不会因抽样而被漏掉。非valid事件按
`role × pilot_state × system_trigger`分成12层，目标样本量240；小层先保证至少10个或直接全量，
剩余额度按层容量比例分配，各层再执行SRSWOR。选择种子为6703。

每个样本写入 `N_h`、`n_h`、纳入概率 `n_h/N_h` 和设计权重 `N_h/n_h`。聚合使用
Horvitz–Thompson事件权重估计总体事件数；episode支持只数实际观察到的独立episode，不用权重膨胀。
这修复了第一版各层等额抽3个却直接把9/30当总体占比，以及第二版episode轮转抽样却误用
`n_h/N_h`等概率权重的两个问题。

第一轮和第二轮扫描全部34张联系表及15张non-valid专用联系表。每格左侧为原始帧，右侧为Day66
叠加，标题包含复核序号、episode、帧号、状态和触发器。对草地/围栏、严重遮挡和异常相机朝向
等41个事件，再生成前后12帧原图/叠加并排短片做第三轮重点确认。

证据等级固定为 `MODEL_ASSISTED_REVIEW_DEVELOPMENT_ONLY`。这不是双人独立标注；“两轮”指
同一模型审查者的重复复核，所以不能声称一致性或真实准确率。

## 5. 视觉复核结果

| 主视觉标签 | 已复核事件 | 加权总体事件估计 | 已观察episode |
|---|---:|---:|---:|
| `no_failure_visible` | 389 | 2,538.44 | 25 |
| `state_machine_confirmation_only` | 68 | 1,061.35 | 22 |
| `central_row_occupancy` | 35 | 645.75 | 18 |
| `severe_occlusion` | 35 | 521.65 | 14 |
| `suspected_ood_valid` | 6 | 6.00 | 1 |
| `illumination_or_glare` | 3 | 35.81 | 3 |

`no_failure_visible`不是正确真值，只表示在当前缩略图和上下文中没有发现明显错误。最严重的
单点问题是同一episode末尾进入草地/围栏后仍出现6个valid事件；因为只来自1个episode，Day67
没有把它包装成跨场景主因。

## 6. Day68选择规则与结果

第三版门禁在看新增样本标签前固定为：至少5个已复核事件、至少3个独立episode、在“可行动失败事件”中占比
至少10%；排序先看严重度，再看episode覆盖，最后看事件占比。原始帧数不参与优先级选择。

加权可行动失败总体估计为1,209.21个事件。`severe_occlusion`估计521.65个，占43.14%，实际复核
35个并覆盖14个episode，严重度4，因此通过并成为Day68唯一目标。`suspected_ood_valid`严重度更高，但全部集中在一个episode，未过
跨episode门；它作为高风险哨兵保留，Day68不得借此调到该单段视频。

Day68允许研究的方向是“遮挡可观测性/拒绝”，不是提高valid比例。验收必须首先保持非valid
导航泄漏为0，并针对冻结的Day67严重遮挡事件检查是否减少疑似不安全valid；若只能通过普遍拒绝
取得改善，应报告覆盖率代价。

## 7. TDD与优化记录

本日新增测试均先看到预期失败，再写最小实现。初版把所有track pair变化都作为事件边界，
导致非导航失败被过度碎片化；新增回归测试后，只在身份敏感状态中使用pair。另一轮初始Day68
门错误地用全部332个复核事件作占比分母，因296个valid被强制纳入而稀释失败类别；第一版虽改用
30个可行动失败事件，仍忽略各抽样层总体大小不同。第二版虽然扩大到240个non-valid样本，却把
episode轮转选择错当成层内等概率抽样。第三版改成真正的层内SRSWOR，并要求240个non-valid事件
逐个显式列入视觉决定；第一、二版结果保留但明确作废，不再支持Day68决策。

Day67共25项测试通过；Day61～67联合测试在最终验收中执行。

## 8. 复现命令

```powershell
& 'D:\conda\envs\forest-species\python.exe' -X utf8 `
  'D:\opencv-learning\67_crop_row_failure_taxonomy\code\day67_failure_taxonomy.py' `
  --day66-output 'D:\DL_code\data\crop_row_perception\day66_offline_video_pilot' `
  --day66-result 'D:\DL_code\data\crop_row_perception\day66_offline_video_pilot\day66_results.json' `
  --manifest 'D:\DL_code\data\crop_row_perception\sources\lecrop_data\crow_video_manifest.json' `
  --output-dir 'D:\DL_code\data\crop_row_perception\day67_failure_taxonomy' `
  --review-decisions 'D:\DL_code\data\crop_row_perception\day67_failure_taxonomy\day67_review_decisions_v3.json'
```

首次生成联系表时额外添加 `--render-review-packet`。复核决定是本地证据文件，不随仓库分发。

## 9. 文件地图与边界

- `day67_failure_taxonomy.py`：输入隔离、事件化、分层抽样、复核渲染、标注校验、聚合和决策门；
- `day67_taxonomy.json`：冻结标签和Day68选择规则；
- `test_day67_failure_taxonomy.py`：协议、事件、冻结隔离、渲染、标注、聚合与决策测试；
- 本地 `day67_results_v3.json`：正式数字和输入/协议/复核决定哈希；
- 本地 `review_sheets_v3/`、`review_nonvalid_v3/`、`review_clips_v3/`：不上传的可视证据；
- 本地第一、二版文件：未加权与权重错配结果，仅作为方法失败记录。

六段 `frozen_same_source_holdout` 只在manifest中计数，未被打开、解码、推理、渲染或标注。
Day69前不得访问。没有独立外部负样本、帧级走廊真值、相机/车体标定和实车试验，因此当前成果
仍是同源开发范围内的离线失败审查，不是机器人安全导航证明。
