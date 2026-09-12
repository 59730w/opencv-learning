# Day71：作物行 Pilot 的 CPU/CUDA 分层一致性诊断与稳定执行策略

日期：2026-09-12

## 1. 今天解决什么问题

Day70 在同一段89帧开发视频上发现：生产CPU路径得到39个 `valid`，默认CUDA路径只有26个
`valid`，共有37帧最终状态和导航可用性不一致。Day71不修改模型、阈值、解码协议、时序配置、
Day68遮挡守卫或Day69冻结外部结果，而是把设备、执行后端、batch形状和数值精度逐项拆开。

今天的核心目标不是让浮点数组逐位相等，而是复现Day70失败、定位第一个受控原因，并找到不改变
模型语义且不牺牲现有CUDA batch吞吐的稳定执行策略。

## 2. 冻结边界

- 输入仍是Day70使用的同字节89帧 `shifted_development` 视频，SHA-256为
  `b9da6a1575659cdf8478764736fbf26bf14b7aed4b69bafa20d2b7b4a0095968`；
- Day63权重、Day65结果、Day66时序配置和Day68遮挡配置在实验前后哈希一致；
- 没有读取SSR图像，没有根据SSR Recall 0.7143调参；
- Day69和Day70文件没有被改写；
- 本次只形成开发期复现证据，不形成外部泛化、安全率或机器人控制证据。

## 3. 环境与计时边界

- Python 3.9.25；
- PyTorch 2.5.1+cu121，CUDA 12.1，cuDNN 9.1；
- NVIDIA GeForce RTX 3050 Laptop GPU；
- `torch.backends.cudnn.benchmark=False`；
- 原始CUDA路径 `torch.backends.cudnn.allow_tf32=True`；
- 每条路径先预热，再重复5次；各路径重复推理的概率数组均逐元素一致。

诊断时间只包含张量转换、模型推理、sigmoid与结果回到CPU，不包含视频解码、四通道特征预处理、
行解码、光流、Kalman、状态机、遮挡守卫和视频编码。因此下表只能用于本次后端诊断，不能当作
完整Pilot端到端速度。

## 4. 六臂受控实验

所有路径接收同一个 `[89, 4, 192, 192] uint8` 四通道特征数组。

| 路径 | 设备与执行方式 | TF32 | Batch | 中位诊断时间/帧 | 最终状态 |
|---|---|---:|---:|---:|---|
| CPU eager | CPU eager | 不适用 | 1 | 30.365 ms | 39 valid / 1 candidate / 49 degraded |
| 生产CPU | TorchScript optimize + channels-last | 不适用 | 1 | 20.122 ms | 39 / 1 / 49 |
| CUDA batch1 | CUDA eager | 开 | 1 | 4.981 ms | 33 / 1 / 55 |
| Day70 CUDA | CUDA eager | 开 | 32 | 2.089 ms | 26 / 1 / 62 |
| CUDA无TF32 batch1 | CUDA eager | 关 | 1 | 4.629 ms | 39 / 1 / 49 |
| 稳定CUDA | CUDA eager | 关 | 32 | 2.059 ms | 39 / 1 / 49 |

六条路径均为0个导航合同违规。

## 5. 根因证据

| 受控比较 | 状态差异帧 | 导航差异帧 | 说明 |
|---|---:|---:|---|
| CPU eager vs 生产CPU | 0 | 0 | TorchScript优化不是Day70最终状态差异来源 |
| CPU eager vs CUDA TF32 batch1 | 14 | 14 | 设备/TF32数值路径已能改变最终状态 |
| CUDA TF32 batch1 vs batch32 | 31 | 31 | TF32开启时，batch形状进一步改变执行数值路径 |
| 生产CPU vs Day70 CUDA TF32 batch32 | 37 | 37 | 精确复现Day70失败 |
| CPU eager vs CUDA无TF32 batch1 | 0 | 0 | 关闭TF32后，batch1最终状态恢复一致 |
| CPU eager vs CUDA无TF32 batch32 | 0 | 0 | 关闭TF32后，batch32仍保持最终一致 |
| 生产CPU vs CUDA无TF32 batch32 | 0 | 0 | 新策略直接匹配现有生产CPU参考 |
| CUDA TF32 batch32 vs CUDA无TF32 batch32 | 37 | 37 | 固定设备、后端和batch后，仅切换TF32即可复现并消除37帧差异 |

用于根因归因的最强直接对照是最后一项：固定CUDA eager和batch32，仅关闭TF32，就让原有37帧
状态/导航差异消失。用于执行策略验收的是倒数第二项“生产CPU vs CUDA无TF32 batch32”：89帧的
最终状态、导航可用性和解码行数量一致。概率数组并非逐位相同，最大绝对差为 `2.8908e-05`、
平均绝对差为 `2.0101e-07`，但没有像素跨过 `0.20` 解码阈值；诊断层最大行置信度差为
`9.2387e-06`，所以差异没有改变离散行结构和时序状态机。真实CLI的连续几何容差结果见第6节。

在这段开发视频和当前软件/硬件版本下，因果结论为
`ROOT_CAUSE_IDENTIFIED: cudnn_tf32_execution_path`：只关闭cuDNN TF32，就同时消除了batch1和
batch32的最终状态差异。该结论不能自动外推到其他GPU、驱动、PyTorch版本或全部视频。

## 6. Day71 稳定运行入口

新增入口不会改写Day69文件。它在调用冻结CLI之前设置
`torch.backends.cudnn.allow_tf32=False`，并强制使用已经验证的CUDA batch32；若调用者显式传入
其他设备或batch，入口会直接拒绝：

```powershell
D:\conda\envs\forest-species\python.exe -X utf8 `
  71_crop_row_device_parity_diagnosis/code/run_crop_row_pilot_device_stable.py `
  --input "D:\你的数据\field.mp4" `
  --device cuda `
  --output-dir "D:\你的输出\field_stable"
```

真实CLI端到端复核已经处理同一89帧视频，得到39 `valid`、1 `candidate`、49 `degraded`、
0 `reject`，导航合同违规为0；输出MP4完整解码89帧。稳定入口还把实际设备、batch和TF32策略
写入 `pilot_report.json`，避免结果文件只声称、却未执行该策略。与真实生产CPU CLI逐帧比较时，
89/89帧状态和导航可用性严格一致，公开几何结构一致；连续几何最大绝对差为
`4.8224e-06`，通过本次开发期诊断中明示采用的 `1e-05` 绝对容差。该容差是在观察本次误差后
形成的工程验收界限，不是预注册阈值，也不能直接外推到新视频或新设备。由于89帧均存在至少一个非零浮点差，
这里不宣称几何逐位相等。完整本地输出位于
`D:/DL_code/data/crop_row_perception/day71_device_parity/stable_cli_run/`。

诊断报告还逐项绑定两侧 `pilot_report.json`：输入视频路径、episode、连续帧号、JSONL路径与
SHA-256、CPU生产后端，以及CUDA设备、batch32和TF32关闭策略均通过，避免把错位帧或旧产物
误当成本次真实CLI证据。

## 7. 产物与复现

- `day71_device_parity.py`：六臂诊断、重复性检查、分层比较、根因和策略门禁；
- `day71_device_parity_result.json`：机器可读结果；
- `day71_frame_differences.csv`：89帧各路径状态、行数和逐帧概率最大差；
- `run_crop_row_pilot_device_stable.py`：设备稳定CLI入口；
- `../tests/test_day71_device_parity.py`：数组证据、逐帧比较、根因判定和运行策略测试。

运行诊断：

```powershell
D:\conda\envs\forest-species\python.exe -X utf8 `
  71_crop_row_device_parity_diagnosis/code/day71_device_parity.py
```

成功标记为 `DAY71_DEVICE_PARITY_DIAGNOSIS_COMPLETE`。

## 8. 今天学到的关键点

1. 同一路径重复多次完全一致，只能证明路径内可重复，不能证明跨后端一致；
2. 神经网络概率图中的小差异会被阈值、二值连通结构、行方向估计和时序状态机非线性放大；
3. batch大小不仅影响速度，也可能改变cuDNN算法和TF32数值轨迹；
4. “最终状态一致”不等于“浮点概率逐位一致”，应分别报告；
5. 最好的工程修复不是调阈值掩盖差异，而是控制引起差异的运行时精度变量。

## 9. Day71结论与剩余边界

Day71核心门为 `PASS_DEVELOPMENT_REPRODUCIBILITY`：Day70的37帧失败已被精确复现；在当前环境、
同一89帧开发视频上，关闭cuDNN TF32的CUDA eager batch32与现有生产CPU CLI实现89/89帧最终
状态和导航可用性严格一致，连续几何误差小于 `1e-05`，同时保留batch32诊断吞吐。

这不是完整跨设备泛化证明。稳定策略还没有在全部25段开发视频、6段同源冻结视频、其他GPU或新
独立视频上验收，因此“任意设备严格一致”仍不可声明。SSR Recall 0.7143、完整多行外部泛化、
目标域拒识、真实视频安全率、米制定位和机器人闭环控制均没有被Day71解除。

Day72最合理的下一步，是先在25段开发视频上验证该执行策略；冻结视频是否再次用于纯工程复现，
应在导师讨论后明确协议再决定，不能借设备修复重新解释Day69冻结准确率。
