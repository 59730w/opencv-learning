# Day58：ONNX Runtime CPU 线程控制与 INT8 性能诊断

## 1. 今天解决什么问题

Day57 得到一个必须保留的负结果：QDQ S8S8 INT8 模型体积缩小了74.719%，内部质量门禁也通过，但在当前 Windows CPU 上的稳态推理反而慢于 FP32。

Day58 不再增加另一种量化格式，也不同时修改运行时版本、图结构或校准方案。今天只回答一个更窄的问题：

> ONNX Runtime 默认的算子内线程配置，是否足以解释 Day57 中 INT8 相对 FP32 的性能下降？

这一步符合当前学习路线：Day55 建立可信基准，Day56 学习 profiling，Day57 完成静态量化，Day58 则学习如何把负结果拆成单变量、可证伪的性能实验。

## 2. 先区分三个概念

### 2.1 模型更小

INT8 权重占用更少磁盘和内存。它回答的是“模型体积是否降低”，不直接回答“推理是否更快”。

### 2.2 INT8 相对 FP32 更快

在同一个线程配置和 Batch 下，定义：

```text
INT8 相对 FP32 加速比 = FP32 median_ms / INT8 median_ms
```

- 大于1：INT8 相对更快；
- 等于1：中位延迟相同；
- 小于1：INT8 相对更慢。

### 2.3 某个线程配置绝对更快

对同一个模型和 Batch，定义：

```text
默认线程相对单线程比值 = 默认线程 median_ms / 单线程 median_ms
```

大于1才表示单线程的绝对延迟更低。不能因为“单线程下 INT8 比 FP32 快”，就误写成“单线程是更好的部署配置”。

## 3. 实验假设与预注册判断

### 3.1 假设

ORT 默认 intra-op 线程调度可能让 FP32 卷积获得比当前 QDQ S8S8 INT8 图更高的并行收益，导致 INT8 在默认线程下相对落后。

### 3.2 唯一自变量

```text
intra_op_num_threads = 0  # ONNX Runtime 默认配置
intra_op_num_threads = 1  # 显式单线程
```

### 3.3 固定变量

- FP32 与 Day57 QDQ S8S8 INT8 模型不变；
- ONNX Runtime 1.19.2 CPUExecutionProvider；
- `ORT_SEQUENTIAL`；
- `ORT_ENABLE_ALL`；
- `inter_op_num_threads=1`；
- `cv::setNumThreads(1)`；
- profiling关闭；
- 只计时 `Session::Run`；
- Batch固定为1和6；
- 六张输入图片及顺序完全相同；
- 每组预热5次，正式测量30次；
- 以中位数作为主要指标，同时报告P90和吞吐量。

### 3.4 预注册解释规则

将2%设为有实际意义的相对改善阈值：

- `full_support`：两个Batch在单线程下都使 INT8 加速比达到至少1.0，并且相对默认线程至少改善2%；
- `partial_support`：至少一个Batch改善2%，但不满足全部条件；
- `not_supported`：两个Batch都没有改善2%。

规则在运行真实基准前写入 `thread_experiment_contract.json`，避免看到结果后再调整成功定义。

## 4. 当前实验环境

| 项目 | 实测/固定值 |
|---|---|
| CPU | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| 逻辑处理器 | 12 |
| 操作系统 | Windows x64 |
| ONNX Runtime DLL | 1.19.20240830.5.ffceed9 |
| Execution Provider | CPUExecutionProvider |
| FP32模型 | Day53动态Batch ResNet18 ONNX |
| INT8模型 | Day57 QDQ、S8S8、per-channel静态PTQ模型 |
| C++计时器 | Day55 Release `day55_benchmark.exe` |

这是一台机器、一个运行时版本和一张模型图上的诊断结论，不外推到其他CPU、ORT版本、QOperator、U8U8、GPU、TensorRT或嵌入式设备。

## 5. 为什么复用 Day55 C++ 基准

Day58 没有重新编写推理和计时循环，而是让 Python 编排器调用 Day55 已验收的 C++ 程序。这样能够保持：

- 相同的 OpenCV C++ 预处理；
- 相同的 ONNX Runtime Session 配置入口；
- 相同的 `steady_clock` 计时；
- 相同的 `Session::Run` 边界；
- 相同的CSV和logits输出格式。

Day58 新增的职责只有：生成八组合实验矩阵、运行和汇总、核对原始计数、执行正确性门禁、计算派生比值并输出证据JSON。

## 6. 测试驱动实现

### 6.1 RED

先编写6项行为测试，其中最初5项因实现文件不存在而失败：

1. 八组合矩阵完整且无重复；
2. 比值只接受正延迟；
3. 命令固定为runtime-only、5次预热和30次测量；
4. 同模型跨线程logits必须有限且ordered Top-3一致；
5. 派生加速比和线程比值方向正确；
6. FP32/INT8成对运行，并平衡每一对中先运行的模型。

### 6.2 GREEN

实现最小编排器后，Day58的6项测试全部通过；Day56–58合计12项相关pytest随后全部通过，Day55 C++基准契约另行执行并通过。

### 6.3 一次实验设计修正

第一版执行顺序先跑完全部FP32，再跑全部INT8，可能把温度、系统负载或时间漂移混入模型比较。第一次计时因此不作为最终结果。

随后先写一个会失败的顺序测试，再把最终顺序改成按相同Batch和线程成对执行 FP32/INT8，并平衡每对中哪个模型先运行。本文只报告修正后的最终测量。

## 7. 正确性门禁

性能解释前，分别比较每个模型在 `intra=0` 与 `intra=1` 下的最后一次输出：

| 模型 | Batch | Logits数量 | 平均绝对差 | 最大绝对差 | Ordered Top-3 |
|---|---:|---:|---:|---:|---|
| FP32 | 1 | 50 | 0 | 0 | 一致 |
| FP32 | 6 | 300 | 0 | 0 | 一致 |
| INT8 | 1 | 50 | 0 | 0 | 一致 |
| INT8 | 6 | 300 | 0 | 0 | 一致 |

因此线程切换没有改变本次输入上的输出，性能比较通过正确性门禁：

```text
DAY58_CORRECTNESS_OK
```

这里的“一致”只证明相同模型跨线程配置的一致性，不代表 FP32 与 INT8 logits 数值等价，也不代表分类准确率或外部泛化改善。

## 8. 最终性能结果

共8个组合，每组30次正式测量，保留240条原始计时。

下表是本次交付冻结的定量快照，跟踪在 `results/day58_benchmark_snapshot.json`；机器相关raw CSV和logits仍留在Git忽略的 `outputs/`。以后重跑要求复现矩阵、正确性门禁和“默认线程慢、单线程相对排序反转”的方向，不要求毫秒值逐位相同。

| 模型 | Batch | Intra-op | Median | P90 | 吞吐量 | 相对同线程FP32 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1 | 默认(0) | 8.5475 ms | 9.9572 ms | 123.303 img/s | 1.000× |
| INT8 | 1 | 默认(0) | 10.4401 ms | 11.1513 ms | 97.559 img/s | 0.819× |
| FP32 | 1 | 单线程(1) | 27.7338 ms | 28.2164 ms | 35.642 img/s | 1.000× |
| INT8 | 1 | 单线程(1) | 24.7728 ms | 25.0642 ms | 40.304 img/s | 1.120× |
| FP32 | 6 | 默认(0) | 45.1609 ms | 53.8352 ms | 137.085 img/s | 1.000× |
| INT8 | 6 | 默认(0) | 54.4501 ms | 57.6362 ms | 110.669 img/s | 0.829× |
| FP32 | 6 | 单线程(1) | 163.2074 ms | 164.8864 ms | 36.711 img/s | 1.000× |
| INT8 | 6 | 单线程(1) | 146.0713 ms | 147.7512 ms | 40.788 img/s | 1.117× |

### 8.1 默认线程下复现 Day57 方向

- Batch1：INT8中位延迟比FP32高22.142%，加速比0.819×；
- Batch6：INT8中位延迟比FP32高20.569%，加速比0.829×。

绝对数值会受当时系统状态影响，不要求与Day57逐毫秒相同；两个Batch中“INT8慢于FP32”的方向再次复现。

### 8.2 单线程下相对排序反转

- Batch1：INT8比同条件FP32低10.676%延迟，相对加速1.120×；
- Batch6：INT8比同条件FP32低10.500%延迟，相对加速1.117×。

相对于默认线程下的INT8加速比，单线程的相对改善为：

- Batch1：36.741%；
- Batch6：34.714%。

两个Batch都超过预注册2%阈值，而且单线程INT8加速比均超过1.0，因此自动解释为：

```text
Interpretation: full_support
```

## 9. 正确解释：相对证据支持，绝对性能不支持

### 9.1 得到支持的结论

在本机、ORT 1.19.2、当前ResNet18 QDQ S8S8图和CPU EP上，默认intra-op并行对FP32的收益显著高于INT8。关闭算子内并行后，INT8相对FP32的排序在两个Batch中都反转。

因此，**FP32与INT8不同的线程扩展效率足以解释本次受控条件下 Day57 的相对性能下降**。

### 9.2 不能把单线程推荐为优化

与各自的默认线程相比，显式单线程的绝对中位延迟为：

| 模型 | Batch | 单线程相对默认线程 |
|---|---:|---:|
| FP32 | 1 | 慢3.245× |
| INT8 | 1 | 慢2.373× |
| FP32 | 6 | 慢3.614× |
| INT8 | 6 | 慢2.683× |

所以单线程只是诊断探针，不是部署优化。若当前目标是最低稳态CPU延迟，本次八组合中仍应选择默认线程FP32，而不是单线程INT8。

### 9.3 尚未被证明的机制

本实验识别的是“线程扩展效率不同”，没有进一步证明差异具体来自：

- 哪个Conv或Gemm内核；
- QDQ边界的量化/反量化开销；
- S8S8指令与内核实现；
- 内存带宽、缓存或调度粒度；
- ORT新版本是否会改变结论。

要回答这些问题，需要后续独立实验，不能从Day58的数据直接推断。

## 10. 与外部泛化的边界

Day58完全没有重新选择模型、阈值、校准数据或外部样本。它只研究相同模型的CPU线程性能，因此：

- 不改变Day57内部质量门禁结论；
- 不证明量化改善真实森林泛化；
- 不修复负样本被强制分类为已知树种的问题；
- 不形成生产部署就绪结论。

## 11. 复现命令

```powershell
Set-Location D:\opencv-learning

$day58Python = 'D:\conda\envs\forest-species\python.exe'
$day55Exe = 'D:\opencv-learning\55_onnxruntime_cpp_benchmark\build\Release\day55_benchmark.exe'
$fp32Model = 'D:\opencv-learning\53_pytorch_onnx_export\artifacts\forest_species_resnet18.onnx'
$int8Model = 'D:\opencv-learning\57_onnxruntime_static_int8_quantization\artifacts\forest_species_resnet18_int8_qdq.onnx'

& $day58Python 58_onnxruntime_cpu_thread_diagnostic\code\day58_thread_experiment.py `
  --executable $day55Exe `
  --fp32 $fp32Model `
  --int8 $int8Model `
  --class-map 53_pytorch_onnx_export\assets\class_to_idx.json `
  --images `
    55_onnxruntime_cpp_benchmark\assets\banana_bird_256x256.png `
    55_onnxruntime_cpp_benchmark\assets\hotdog_wrap_398x296.png `
    55_onnxruntime_cpp_benchmark\assets\fruit_strip_140x60.png `
    55_onnxruntime_cpp_benchmark\assets\dog_grooming_400x301.jpg `
    55_onnxruntime_cpp_benchmark\assets\cifar_airplane_32x32.png `
    55_onnxruntime_cpp_benchmark\assets\hotdog_pickle_614x419.png `
  --output-dir 58_onnxruntime_cpu_thread_diagnostic\outputs\benchmark `
  --combined-output 58_onnxruntime_cpu_thread_diagnostic\outputs\thread_comparison.csv `
  --evidence-output 58_onnxruntime_cpu_thread_diagnostic\outputs\thread_evidence.json
```

验收实验合同：

```powershell
& $day58Python 58_onnxruntime_cpu_thread_diagnostic\tests\day58_contract.py `
  --lesson-root 58_onnxruntime_cpu_thread_diagnostic `
  --combined-output 58_onnxruntime_cpu_thread_diagnostic\outputs\thread_comparison.csv `
  --evidence-output 58_onnxruntime_cpu_thread_diagnostic\outputs\thread_evidence.json `
  --recorded-result 58_onnxruntime_cpu_thread_diagnostic\results\day58_benchmark_snapshot.json
```

预期标志：

```text
DAY58_CORRECTNESS_OK
DAY58_THREAD_EXPERIMENT_OK
DAY58_CONTRACT_OK
```

## 12. 文件结构

```text
58_onnxruntime_cpu_thread_diagnostic/
├── .gitignore
├── assets/
│   └── README.md
├── code/
│   ├── day58_notes.md
│   └── day58_thread_experiment.py
├── tests/
│   ├── day58_contract.py
│   └── test_day58_thread_experiment.py
├── results/
│   └── day58_benchmark_snapshot.json # 可跟踪的定量快照
├── thread_experiment_contract.json
└── outputs/                         # 机器相关产物，Git忽略
    ├── benchmark/                   # 8组raw/summary/logits
    ├── thread_comparison.csv
    └── thread_evidence.json
```

## 13. 今日学习总结

1. 单变量实验必须同时声明自变量、固定变量和解释阈值。
2. 正确性等价是性能解释的前置门禁。
3. 模型A在单线程下相对模型B更快，不等于单线程绝对更快。
4. CPU推理性能取决于内核和线程扩展效率，INT8不会自动获得与FP32相同的并行收益。
5. 成对、平衡执行可以降低时间漂移对模型比较的混杂。
6. 负性能结果可以通过受控实验转化为可复现、范围明确的知识。

## 14. Day58最终结论

Day58完成了一个预注册、正确性先行、成对平衡的线程诊断实验。结果表明：

> 默认intra-op线程对FP32的绝对加速收益明显高于当前QDQ S8S8 INT8模型；单线程下INT8相对FP32的排序在Batch1和Batch6都反转，因此线程扩展效率差异足以解释本机Day57的相对慢速。但显式单线程使两种模型的绝对延迟都恶化数倍，不能作为部署优化。

当前最佳陈述仍然是：Day57 INT8模型适合作为“体积压缩成功、内部质量可接受”的实验产物；若目标是当前机器的最低CPU延迟，现有证据仍支持默认线程FP32。Day58不改变外部泛化失败和开放集失败的结论。
