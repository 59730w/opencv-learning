# Day57：ONNX Runtime 静态 INT8 量化与联合验收

## 1. 今日目标

Day56 的 Profiling 证明，当前 ResNet18 CPU 推理约98%～99%的算子时间集中在卷积。Day57 因此不盲目尝试动态量化，而是学习能够覆盖 Conv 的静态训练后量化（PTQ），并同时检查：

- 模型是否真正生成 INT8/QDQ 结构；
- 内部 Accuracy 与 Macro-F1 是否保持；
- 冻结外部表现是否被诚实保留；
- C++ CPU 推理是否真的变快；
- 模型体积是否下降。

今天的结论是：**模型压缩和内部质量门禁通过，但当前机器上的 C++ CPU 推理速度反而下降，因此不能把本次 INT8 模型推荐为延迟优化方案。**

## 2. 路线衔接

```text
Day53：导出动态批次 FP32 ONNX
  → Day54：C++ ONNX Runtime 正确推理
  → Day55：关闭 Profiling 的正式性能基准
  → Day56：定位 Conv 与线程热点
  → Day57：静态 INT8 量化，联合检查体积、质量、外部边界和速度
```

Day57 没有重训模型，也没有改变类别体系或预处理契约。

## 3. 阶段一：环境、数据和实验契约

### 3.1 环境

| 组件 | 版本/状态 |
|---|---|
| Python | 3.9.25 |
| OpenCV | 4.10.0 |
| NumPy | 1.23.5 |
| ONNX | 1.16.1 |
| ONNX Runtime | 1.19.2 |
| `quantize_static` | 可用 |
| `CalibrationDataReader` | 可用 |
| QDQ、QInt8 | 可用 |
| 新下载 | 无 |

### 3.2 FP32 参考模型

| 项目 | 值 |
|---|---|
| 文件 | Day53 `forest_species_resnet18.onnx` |
| SHA-256 | `9403c2db6abec8a851fc96bc0053951a9bd3dc79d1d1fa82dab4089d7b23bc8c` |
| 大小 | 44,802,759 bytes |
| 输入 | `images [batch,3,224,224] float32` |
| 输出 | `logits [batch,50] float32` |
| Conv | 20 |
| Gemm | 1 |

### 3.3 校准数据门禁

仓库与实际训练工作区的 `split_manifest.csv` SHA-256 一致。清单包含：

| 划分 | 图片数 | 用途 |
|---|---:|---|
| train | 3891 | 允许抽取校准集 |
| validation | 844 | 量化质量首个门禁，禁止校准 |
| test | 823 | validation 通过后评估，禁止校准 |
| external | 26 | 方案冻结后只比较一次，禁止校准和调参 |

校准集按 `class_index` 和 `relative_path` 确定性排序，每类取前4张，共200张。训练集每类至少56张，因此50类都能覆盖。校准读取器再次断言所有记录的 `split` 必须为 `train`。

### 3.4 量化契约

```text
方法：static PTQ
格式：QDQ
激活：QInt8
权重：QInt8
权重：per-channel
校准：MinMax
目标算子：Conv、Gemm
```

Validation Accuracy 和 Macro-F1 相对 FP32 的下降上限均固定为1个百分点。门禁失败时不得运行 test、external 和性能比较。

## 4. 阶段二：测试驱动生成 INT8 模型

### 4.1 RED

先编写4项测试，首次执行时三个实现脚本尚不存在，测试按预期失败。测试覆盖：

1. 校准样本确定性、类别均衡且只来自 train；
2. Python OpenCV 预处理满足 RGB、NCHW、ImageNet 归一化契约；
3. Accuracy 与 Macro-F1 计算；
4. FP32/INT8 延迟加速比方向。

### 4.2 GREEN

实现：

```text
code/day57_quantize.py
code/day57_evaluate.py
code/day57_benchmark.py
```

随后4项测试全部通过。

### 4.3 INT8 结构与体积

| 项目 | FP32 | INT8 QDQ |
|---|---:|---:|
| 文件大小 | 44,802,759 bytes | 11,326,776 bytes |
| 体积减少 | — | 74.719% |
| Conv | 20 | 20 |
| Gemm | 1 | 1 |
| QuantizeLinear | 0 | 39 |
| DequantizeLinear | 0 | 81 |

INT8 模型 SHA-256：

```text
916adf8b7341797cdfe7ecf36dcad43e7e1b8e18e282d35c5fe5cb85f052f565
```

ONNX checker 通过，ONNX Runtime CPU Session 可以加载并运行。

## 5. 阶段三：内部质量与冻结外部比较

### 5.1 Validation 门禁

| 指标 | FP32 | INT8 | 下降 |
|---|---:|---:|---:|
| Accuracy | 97.986% | 97.630% | 0.355个百分点 |
| Macro-F1 | 97.756% | 97.378% | 0.378个百分点 |

两项下降都小于1个百分点，内部质量门禁通过。

其他一致性证据：

- Top-1 一致率：99.526%；
- 有序 Top-3 一致率：74.289%；
- logits 平均绝对差：0.2914；
- logits 最大绝对差：11.7200。

量化会明显改变 logits 的绝对数值和部分低位排名，所以不能沿用 FP32 运行时转换时的 `1e-6` 数值等价要求。这里用任务指标、Top-1一致率和类别级指标控制质量。

### 5.2 Test 结果

| 指标 | FP32 | INT8 | 下降 |
|---|---:|---:|---:|
| Accuracy | 98.177% | 97.570% | 0.608个百分点 |
| Macro-F1 | 98.014% | 97.372% | 0.642个百分点 |

- Top-1 一致率：99.028%；
- 有序 Top-3 一致率：73.147%；
- logits 平均绝对差：0.2827；
- logits 最大绝对差：4.2896。

### 5.3 冻结外部比较

量化配置通过 validation 并被锁定后，才读取一次已有的26张冻结外部图片。

| 项目 | FP32 | INT8 |
|---|---:|---:|
| 20张正样本 Accuracy | 5% | 10% |
| 6张负样本平均最大置信度 | 0.655 | 0.677 |

全部26张的 FP32/INT8 Top-1 一致率为88.462%，6张负样本的 Top-1 一致率为100%。

INT8 多正确1张正样本不能解释为泛化改善，因为样本只有20张，而且这不是新的独立外部实验。更重要的是，负样本仍被强制分类为已知树种，平均最大置信度还略有上升。原模型的开放世界失败没有解决。

## 6. 阶段四：关闭 Profiling 的 C++ 性能比较

### 6.1 受控条件

- 复用 Day55 C++ `day55_benchmark.exe`；
- `runtime-only`，只计时 `Session::Run`；
- Profiling 关闭；
- CPU Execution Provider；
- ORT 默认 intra-op 线程；
- Batch1 和 Batch6；
- 每组预热3次，正式测量10次；
- FP32/INT8 使用相同输入、类别映射和计时边界。

### 6.2 性能结果

| 模型 | Batch | Median | P90 | 吞吐量 | 相对 FP32 |
|---|---:|---:|---:|---:|---:|
| FP32 | 1 | 9.8831 ms | 9.9865 ms | 104.163 img/s | 1.000× |
| INT8 | 1 | 10.8439 ms | 11.2988 ms | 92.267 img/s | 0.911× |
| FP32 | 6 | 48.5285 ms | 53.3668 ms | 123.744 img/s | 1.000× |
| INT8 | 6 | 57.9571 ms | 58.7733 ms | 105.681 img/s | 0.837× |

INT8 在当前机器、ORT 1.19.2、QDQ S8S8 和当前图结构下没有加速：

- Batch1 中位延迟增加约9.7%；
- Batch6 中位延迟增加约19.4%；
- 两个 Batch 的吞吐量都下降。

Session 创建时间由约87 ms降到约40～44 ms，但本课的稳态推理结论仍以 `Session::Run` 为准。

## 7. 为什么量化后可能更慢

INT8 并不保证任何 CPU、模型和图格式都会更快。本次可能受到以下因素共同影响：

- QDQ 图中存在额外量化/反量化节点；
- 当前 CPU 指令集、ORT 1.19.2 内核和线程调度未必适合这组 QDQ S8S8 Conv；
- ResNet18 的算子形状和 Batch 较小，转换开销可能抵消整数卷积收益；
- “模型更小”主要降低磁盘和内存占用，不等于 `Session::Run` 必然更快。

这些是后续可验证假设，不在 Day57 内同时尝试 QOperator、U8U8、不同校准算法或多套线程参数，避免将实验矩阵无控制扩张。

## 8. 内部与外部结论

### 内部结论

- 校准数据严格来自 train；
- validation 与 test 的 Accuracy/Macro-F1 下降均小于1个百分点；
- 量化没有造成明显内部类别质量崩溃；
- INT8 模型体积减少74.719%。

### 外部结论

- 外部集没有进入校准或量化选择；
- 原模型真实场景准确率仍极差；
- 负样本仍被高置信度强制分类；
- Day57 不能提出真实森林部署就绪的结论。

### 部署建议

当前 INT8 模型可作为“体积压缩成功、质量可接受”的实验产物，但不应作为当前 Windows CPU 环境的延迟优化版本。若部署目标受存储限制，可以继续评估；若目标是降低延迟，应先调查硬件指令支持、QOperator/U8U8 或目标设备后端，再用同一门禁复验。

## 9. 文件结构与产物

```text
57_onnxruntime_static_int8_quantization/
├── .gitignore
├── quantization_contract.json
├── assets/README.md
├── code/
│   ├── day57_quantize.py
│   ├── day57_evaluate.py
│   ├── day57_benchmark.py
│   └── day57_notes.md
└── tests/test_day57_quantization.py
```

以下内容默认忽略，不上传：

```text
artifacts/*.onnx
outputs/quantization_summary.json
outputs/quality_report.json
outputs/benchmark/*.csv
outputs/benchmark/*.bin
outputs/benchmark_comparison.csv
```

## 10. 今日关键记忆点

1. Profiling 找到热点后，再选择能覆盖热点算子的优化方法。
2. 静态量化必须使用代表性的训练数据校准，validation、test 和外部集不能进入校准。
3. 量化不是数值等价转换，应使用 Accuracy、Macro-F1、预测一致率和错误分析验收。
4. 内部质量保持不代表外部泛化改善。
5. 模型缩小不代表推理变快，正式速度必须关闭 Profiling 后重新测量。
6. 负结果也是有效结果：本次 INT8 缩小74.719%，但当前 CPU 延迟变差，因此不能宣称加速成功。
