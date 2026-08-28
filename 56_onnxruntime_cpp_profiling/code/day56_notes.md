# Day56：ONNX Runtime C++ 算子性能分析

## 1. 今天解决的问题

Day55 已经测出了纯 `Session::Run` 和端到端流程的速度，但只知道“整体用了多久”，还不知道时间具体消耗在哪个算子上。Day56 使用 ONNX Runtime 自带的 Profiling 功能生成 JSON 事件，回答以下问题：

1. CPU 推理的主要热点算子是什么；
2. Batch 从1增大到6后，热点结构是否改变；
3. 默认线程和单线程的差距主要来自哪里；
4. 下一步优化应优先针对哪一部分。

Profiling 会增加记录开销，因此今天的数据只用于诊断热点，不能替代 Day55 的正式基准数据。

## 2. 与整个学习路线的关系

当前学习主线已经从 Python 模型训练进入 C++ 部署：

```text
Day53 ONNX 导出与 Python 端一致性
  → Day54 ONNX Runtime C++ 正确推理
  → Day55 受控性能基准
  → Day56 算子与线程热点定位
  → 后续根据证据选择量化、硬件后端或其他优化
```

先定位热点再选优化方法，可以避免没有证据就直接尝试量化、GPU 或 TensorRT。

## 3. 阶段一：环境与实验契约

### 3.1 环境检查

| 项目 | 结果 |
|---|---|
| ONNX Runtime | 1.19.2 CPU 版 |
| C++ Profiling API | `EnableProfiling`、`EndProfilingAllocated` 均存在 |
| OpenCV | 4.10.0 |
| 编译器 | MSVC 19.35，C++17 |
| 构建类型 | Release |
| Python | 3.9.25，用于 JSON 汇总与契约测试 |
| 下载 | 无 |

CMake 配置成功标记：

```text
DAY56_PROFILING_API_OK
```

### 3.2 固定实验配置

只运行以下三组，防止无目的扩大矩阵：

| 配置名 | Batch | ORT intra-op 线程 |
|---|---:|---:|
| `batch1_default` | 1 | 0，使用 ORT 默认值 |
| `batch6_default` | 6 | 0，使用 ORT 默认值 |
| `batch6_single` | 6 | 1，固定单线程 |

共同控制条件：

- CPU Execution Provider；
- `ORT_ENABLE_ALL` 图优化；
- `ORT_SEQUENTIAL` 执行模式；
- inter-op 线程固定为1；
- OpenCV 线程固定为1；
- 每组先用独立的非 Profiling Session 预热3次；
- 再创建 Profiling Session，记录10次 `Session::Run`；
- 输入张量在运行前准备完毕。

独立预热 Session 可以避免把预热推理混入算子汇总。Profiling Session 的初始化事件仍保留在原始 JSON 中，但算子汇总只选择带 `provider` 的 `Node` 事件，排除 Session 事件和零耗时 fence 事件。

## 4. 阶段二：实现与测试

### 4.1 文件结构

```text
56_onnxruntime_cpp_profiling/
├── .gitignore
├── CMakeLists.txt
├── CMakePresets.json
├── profiling_contract.json
├── assets/README.md
├── code/
│   ├── day56_profiling.cpp
│   ├── day56_analyze_profile.py
│   └── day56_notes.md
└── tests/
    ├── day56_profiling_contract.py
    └── test_profile_analysis.py
```

### 4.2 C++ Profiling 程序流程

1. 检查参数、模型和图片；
2. 使用 Day54 的 OpenCV 预处理函数生成连续 `float32` NCHW 张量；
3. 创建非 Profiling Session 完成预热；
4. 在新的 `SessionOptions` 中调用 `EnableProfiling`；
5. 创建 Profiling Session 并执行固定次数推理；
6. 检查输出形状为 `[batch, 50]` 且所有 logits 有限；
7. 调用 `EndProfilingAllocated` 结束记录；
8. 将 ORT 生成的带时间戳文件规范化为指定 JSON 路径；
9. 保存 logits，供参考实现进行一致性验证。

### 4.3 Python 汇总器

汇总器读取 Profiling JSON，只统计：

```text
cat == "Node"
args.op_name 存在
args.provider 存在
dur 是非负数值
```

按 `(op_name, provider)` 分组，输出：事件数、总耗时、平均耗时和算子时间占比。JSON 中的 `dur` 单位为微秒。

### 4.4 测试驱动记录

先创建汇总器单元测试；由于实现文件尚不存在，2项测试按预期失败。完成实现后，单元测试2/2通过。随后 Release 构建成功，完整 CTest 结果为：

```text
day56_profile_analysis_unit   Passed
day56_profiling_contract      Passed
100% tests passed, 0 tests failed out of 2
```

契约测试还验证了：

- Profiling JSON 确实包含 Node 事件；
- CSV 汇总不为空；
- 不允许未约定的 Batch3；
- Batch1 相对 Day54 的 logits 最大误差不超过 `1e-6`；
- 有序 Top-3 完全相同。

## 5. 阶段三：三组正式诊断

### 5.1 算子汇总结果

下表的“每次算子总时间”是10次推理中所有已统计 Node 事件总和除以10，只用于 Profiling 诊断。

| 配置 | 每次算子总时间 | 每图算子时间 | Conv 占比 | MaxPool 占比 |
|---|---:|---:|---:|---:|
| Batch1 默认线程 | 9.9590 ms | 9.9590 ms | 98.328% | 0.845% |
| Batch6 默认线程 | 41.3056 ms | 6.8843 ms | 97.957% | 1.835% |
| Batch6 单线程 | 162.7723 ms | 27.1287 ms | 99.231% | 0.732% |

每次推理固定包含20个带 Provider 的 Conv 事件。对应的 Conv 总耗时为：

| 配置 | 10次 Conv 总耗时 | 单次平均 Conv 总耗时 |
|---|---:|---:|
| Batch1 默认线程 | 97,925 μs | 9.7925 ms |
| Batch6 默认线程 | 404,619 μs | 40.4619 ms |
| Batch6 单线程 | 1,615,201 μs | 161.5201 ms |

### 5.2 结论

1. **卷积是绝对热点。** 三组实验中 Conv 均占约98%或以上，分类头的 Gemm 仅占约0.01%～0.29%。后续优化若只处理 Top-k、类别映射或输出打印，不会显著改善纯推理性能。
2. **默认线程对 Batch6 的卷积很有效。** Batch6 单线程的算子总时间约为默认线程的 `3.94倍`，主要差距来自 Conv。
3. **Batch 提高了单位图片的算子效率。** 在默认线程下，单位图片算子时间由约9.96 ms降到约6.88 ms。这与 Day55 的批处理吞吐趋势一致，但 Profiling 数值本身不作为正式速度结论。
4. **所有算子都运行在 CPUExecutionProvider。** 今天没有 GPU、TensorRT 或其他硬件后端参与。
5. **下一步优化必须保留正确性门禁。** 无论选择线程调优、量化还是硬件后端，都要先比较相同输入的 logits、Top-1和有序 Top-3，再讨论速度。

## 6. 内部正确性与外部泛化边界

### 6.1 内部正确性

Day56 使用与 Day54 相同的模型、OpenCV 预处理和输入图片。Batch6 两种线程配置相对 Day54 的结果为：

```text
max_abs_default = 0
max_abs_single  = 0
ordered Top-3   = 完全一致
DAY56_BATCH6_CORRECTNESS_OK
```

这证明启用 Profiling 和改变线程配置没有改变当前测试输入的推理结果。它属于部署链路内部正确性证据，不等同于模型精度评估。

### 6.2 外部泛化

今天复用 Day55 的6张多来源图片，以覆盖不同尺寸、内容和难度，但这些图片不是森林树种目标分布。模型对非树皮图片仍会强制输出50个已知树种之一，说明前三个实战暴露出的开放世界和外部泛化限制并未被部署优化解决。

因此，Day56只允许提出“运行时热点已定位”的结论，不能提出“模型适合真实森林场景”的结论。未来新项目仍必须同时守住：

- 内部有效性：按最高泄漏单位划分，验证数据、标签、指标和误差分析；
- 外部泛化：目标相关 OOD 开发集和独立冻结外部测试，不能用外部测试调参。

## 7. 生成文件与复现

原始 Profiling JSON、CSV 和 logits 都保存在 `outputs/`，并由 `.gitignore` 排除，不上传 GitHub。源代码、测试、实验契约和中文笔记会被保留。

配置、编译和测试：

```powershell
cmake --preset msvc-x64-release
cmake --build --preset msvc-x64-release
ctest --preset msvc-x64-release
```

环境变量沿用 Day55 的依赖位置，但使用 Day56 专用名称：

```text
ONNXRUNTIME_ROOT=D:\opencv-deps\onnxruntime-win-x64-1.19.2
DAY56_OPENCV_DIR=D:\opencv-deps\opencv-4.10.0-msvc\opencv\build\x64\vc16\lib
DAY56_PYTHON=D:\conda\envs\forest-species\python.exe
```

## 8. 今日关键记忆点

1. Benchmark 回答“多快”，Profiling 回答“时间花在哪里”。
2. Profiling 有额外开销，不能直接替代正式基准。
3. ORT Profiling 需要在创建 Session 前通过 `SessionOptions` 开启，并在结束时显式取回文件名。
4. 解析 JSON 时应区分 Session、Node、kernel 和 fence 事件，不能把全部事件直接相加。
5. 当前 ResNet18 CPU 推理的主要热点是 Conv，而不是 Gemm 或输出后处理。
6. 性能配置变化后仍要重新检查 logits 与排名，内部正确性不能因为追求速度而省略。
7. 部署正确和运行更快都不会自动改善模型的外部泛化。
