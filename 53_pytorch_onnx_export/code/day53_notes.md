# Day53：将 PyTorch 森林树种模型导出为 ONNX

## 1. 今日定位

Day44-Day52 已完成 C++ OpenCV、CMake、调试、类封装、多线程、测试、安装与打包。今天开始连接两条已经学过的路线：

```text
PyTorch 深度学习模型
        ↓ 导出
ONNX 计算图
        ↓ 加载
ONNX Runtime
        ↓ 后续继续
C++ / OpenCV 图像预处理与推理
```

今天不训练新模型，而是复用实践项目 3 的森林树种 ResNet18，完成以下闭环：

1. 从 PyTorch 检查点重建 50 类 ResNet18；
2. 导出 ONNX opset 18 模型；
3. 用 `onnx.checker` 检查模型结构；
4. 把输入批次轴设置为动态维度；
5. 使用 ONNX Runtime CPU 执行真实图片推理；
6. 比较 PyTorch 与 ONNX Runtime 的 logits、Top-1 和 Top-3；
7. 记录导出器版本兼容性问题和部署边界。

今天的重点不是“生成一个 `.onnx` 文件”，而是证明这个文件结构有效、接口清楚，并且推理结果与原模型足够一致。

---

## 2. 今日完成结果

### 2.1 环境与仓库

- 学习仓库：`D:\opencv-learning`；
- 开始时 `main` 与 `origin/main` 差异为 `0/0`；
- 复用环境：`D:\conda\envs\forest-species\python.exe`；
- Python：3.9.25；
- PyTorch：2.5.1+cu121；
- TorchVision：0.20.1+cu121；
- ONNX：1.16.1；
- ONNX Runtime：1.19.2；
- ONNXScript：0.1.0。

在安装 ONNX 依赖前已保存 `pip freeze` 快照并执行安装预演。最终只新增 9 个包：

```text
onnx==1.16.1
onnxruntime==1.19.2
onnxscript==0.1.0
protobuf==6.33.6
coloredlogs==15.0.1
humanfriendly==10.0
flatbuffers==25.12.19
ml_dtypes==0.5.4
pyreadline3==3.5.6
```

PyTorch、TorchVision、NumPy、OpenCV、Gradio、Pydantic 等原有包版本没有变化。`pip check` 没有发现损坏的依赖，原森林 Demo 的 10 项回归测试全部通过，同一张 Acacia 图片在安装前后都得到 100% 的 Top-1 结果。

### 2.2 ONNX 导出结果

```text
模型：forest_species_resnet18.onnx
检查点：best.pt，第 13 轮，finetune 阶段
类别数：50
参数量：11,202,162
ONNX 节点数：49
ONNX opset：18
文件大小：44,802,759 bytes（约 42.73 MiB）
SHA256：9403c2db6abec8a851fc96bc0053951a9bd3dc79d1d1fa82dab4089d7b23bc8c
```

模型接口：

```text
输入名称：images
输入形状：[batch, 3, 224, 224]
输入类型：float32

输出名称：logits
输出形状：[batch, 50]
输出含义：每张图片对 50 个已知类别的未归一化分数
```

确定性成功标记：

```text
DAY53_ONNX_EXPORT_OK
```

### 2.3 一致性验证结果

实际运行了 batch=1、batch=2、batch=3 三种输入规模，全部通过。batch=3 的主要结果如下：

```text
输入形状：(3, 3, 224, 224)
输出形状：(3, 50)
执行后端：CPUExecutionProvider
最大绝对误差：0.00000763
平均绝对误差：0.00000235
允许误差：atol=1e-4，rtol=1e-4
全部 logits：通过 np.allclose
Top-1：PyTorch 与 ONNX Runtime 完全一致
Top-3：PyTorch 与 ONNX Runtime 完全一致
```

三张图片的 ONNX Top-3：

| 图片 | Top-1 | Top-2 | Top-3 |
| --- | --- | --- | --- |
| `acacia_IMG_6348.JPG` | Acacia | Annona squamosa | Prunus salicina |
| `adenanthera_microsperma_IMG_5777.JPG` | Adenanthera microsperma | Magnolia alba | Polyalthia longifolia |
| `cananga_odorata_IMG_5483.JPG` | Cananga odorata | Tamarindus indica | Cedrus |

确定性成功标记：

```text
DAY53_ONNX_COMPARE_OK
```

---

## 3. 目录结构

```text
53_pytorch_onnx_export/
├── .gitignore
├── assets/
│   ├── README.md
│   ├── acacia_IMG_6348.JPG
│   ├── adenanthera_microsperma_IMG_5777.JPG
│   ├── cananga_odorata_IMG_5483.JPG
│   └── class_to_idx.json
└── code/
    ├── day53_common.py
    ├── day53_export_onnx.py
    ├── day53_compare_inference.py
    ├── requirements_day53.txt
    └── day53_notes.md
```

运行后还会出现：

```text
53_pytorch_onnx_export/
├── artifacts/
│   └── forest_species_resnet18.onnx
└── outputs/
    ├── export_summary.json
    ├── comparison_batch1.json
    ├── comparison_batch2.json
    └── comparison_report.json
```

`artifacts/`、`outputs/` 和 `__pycache__/` 已被本目录 `.gitignore` 忽略。因此约 42.73 MiB 的 ONNX 文件、机器相关绝对路径和临时运行报告不会上传 GitHub。仓库保留代码和小型示例资产，使用者需要从原森林项目的 `best.pt` 重新导出模型。

---

## 4. ONNX 到底解决什么问题

PyTorch 的 `.pt` 检查点通常保存参数、优化器状态或训练配置。它与 PyTorch 的 Python 模型定义紧密相关。ONNX 则把推理过程表示成一种跨框架的计算图：

```text
输入张量
  ↓
Conv / BatchNormalization / ReLU / Add / GlobalAveragePool / Gemm
  ↓
输出张量
```

ONNX 的主要价值是把“模型训练框架”和“模型部署运行时”分开：

- 训练时仍可使用 PyTorch；
- 部署时可以使用 Python、C++、C# 等语言中的 ONNX Runtime；
- 后续也可以把 ONNX 交给 TensorRT 或其他推理工具继续优化；
- 不需要在 C++ 中重新实现完整的 PyTorch 网络定义。

但 ONNX 不是完整应用。它只包含网络计算图和权重，通常不自动包含图片读取、BGR/RGB 转换、缩放裁剪、归一化、类别名称、Top-k 展示和未知类别拒绝逻辑。

---

## 5. 五个必须分清的对象

### 5.1 检查点 `best.pt`

本项目的检查点是一个 Python 字典，主要包含：

```text
model_state_dict
config
epoch
stage
validation_metrics
```

其中 `model_state_dict` 只有参数，不能脱离正确的 ResNet18 结构直接推理。因此导出前必须先创建 50 类 ResNet18，再严格加载参数。

### 5.2 PyTorch 模型

`day53_common.py` 重建标准 ResNet18，把最后的全连接层替换成 50 类输出：

```python
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 50)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
```

`weights=None` 很重要：此时不是重新下载 ImageNet 权重，而是准备一个结构相同的空模型，再加载已经训练好的森林分类参数。

`strict=True` 要求参数名称和形状完全对应。若模型结构与检查点不一致，应立即失败，而不是悄悄忽略参数。

### 5.3 示例输入

导出器需要一个形状和类型正确的输入来追踪网络：

```python
example_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
```

它不是训练数据，也不会改变模型参数。它只告诉导出器：“实际输入是一批 NCHW 排列的 float32 图片张量”。

### 5.4 ONNX 模型

ONNX 文件保存网络节点、张量连接关系、权重、输入输出接口和少量元数据。今天额外写入了：

- 课程名称；
- 任务类型；
- 预处理规则；
- 类别数；
- 检查点轮次；
- 闭集模型限制。

### 5.5 ONNX Runtime Session

`InferenceSession` 负责加载并执行 ONNX：

```python
session = ort.InferenceSession(
    str(onnx_model),
    providers=["CPUExecutionProvider"],
)
```

今天安装的是 CPU 版 `onnxruntime`，因此明确使用 `CPUExecutionProvider`。森林环境中 PyTorch 的 CUDA 仍然可用，但这不代表 CPU 版 ONNX Runtime 会自动使用显卡。若以后需要 CUDA Provider，必须单独评估并安装匹配的 `onnxruntime-gpu`，不能同时假设两套 GPU 依赖天然兼容。

---

## 6. 输入预处理为什么必须完全一致

原森林模型的验证预处理是：

```text
PIL 读取图片
→ 转 RGB
→ Resize(256)
→ CenterCrop(224)
→ ToTensor，像素从 0～255 变成 0～1
→ ImageNet mean/std Normalize
→ NCHW float32
```

归一化公式是：

```text
normalized = (pixel - mean) / std
```

三个通道分别使用：

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

这部分目前在 ONNX 模型外完成。只要 C++ 端出现以下任意差异，即使 ONNX 本身完全正确，结果也会明显变化：

- OpenCV 读入后没有从 BGR 转换成 RGB；
- 直接拉伸到 224×224，而不是先按短边缩放到 256 再中心裁剪；
- 忘记除以 255；
- mean/std 顺序错误；
- HWC 没有转换成 CHW；
- 使用 `double` 或 `uint8`，而模型需要 float32；
- 内存排列与 `[batch,3,224,224]` 不一致。

因此后续 C++ 推理最重要的验收，不只是“程序可以运行”，而是让 C++ 预处理张量和 Python 预处理张量可比较。

---

## 7. 静态批次与动态批次

若不声明动态轴，用形状 `[1,3,224,224]` 导出后，模型往往只能接收 batch=1。今天明确声明：

```python
dynamic_axes={
    "images": {0: "batch"},
    "logits": {0: "batch"},
}
```

因此最终接口变成：

```text
images: [batch, 3, 224, 224]
logits: [batch, 50]
```

这里的 `batch` 是符号维度，不是字符串数据。它表示运行时可以传入不同数量的图片。今天不是只读取模型结构，而是实际执行了 batch=1、2、3，三个规模都得到成功标记。

空间尺寸 224×224 没有设为动态，这是有意的。ResNet18 理论上能处理某些其他尺寸，但项目训练和验证使用 224×224，保持固定空间尺寸更容易保证预处理一致，也能降低初学部署时的接口复杂度。

---

## 8. 为什么导出 logits，而不把 softmax 放进模型

模型最后输出 `[batch,50]` 的 logits。logits 是进入 softmax 之前的原始分数。

保留 logits 有三个优点：

1. 与训练模型的原始 `forward()` 保持一致；
2. 部署端可以根据需求计算 softmax、Top-k 或其他分数；
3. 比较导出前后数值时，可以直接比较网络原始输出，避免 softmax 压缩差异。

如果需要概率，可以在推理端按类别维执行：

```python
probabilities = torch.softmax(logits, dim=1)
```

注意：softmax 只能把 50 个已知类的分数归一化，并不会赋予模型“未知类别拒绝”能力。对于不属于 BarkVN-50 的图片，闭集模型仍会被迫选出一个已知类别。

---

## 9. 为什么必须比较 PyTorch 与 ONNX Runtime

`onnx.checker.check_model()` 只证明模型格式和图结构满足 ONNX 规则，不能证明数值与 PyTorch 一致。

完整验证需要相同输入经过两条路径：

```text
同一个预处理后的 float32 batch
        ├── PyTorch ResNet18 ──→ pytorch_logits
        └── ONNX Runtime ─────→ onnx_logits
```

然后检查：

1. 输出形状是否相同；
2. 全部 logits 是否在容差内；
3. Top-1 是否相同；
4. Top-3 的类别和顺序是否相同。

浮点推理中的卷积实现、运算融合和累加顺序可能不同，因此不应要求每一个 bit 完全相同。今天使用：

```python
np.allclose(pytorch_logits, onnx_logits, atol=1e-4, rtol=1e-4)
```

其中：

- `atol` 是绝对误差容限；
- `rtol` 是相对于参考值大小的误差容限；
- 实际最大绝对误差约 `7.63e-6`，明显小于 `1e-4`；
- Top-1 和 Top-3 完全一致。

所以可以判定：在今天测试的 CPU 环境和真实样本上，ONNX 模型忠实复现了 PyTorch 模型的推理结果。

---

## 10. 导出器兼容性排查记录

今天先尝试了 PyTorch 2.5 的新 Dynamo ONNX 导出路径：

```python
dynamo=True
dynamic_shapes=...
```

排查过程保留了两个有价值的结果。

### 10.1 batch=1 被特化成常量

使用 batch=1 的示例输入时，Dynamo 报告动态维约束冲突，认为 batch 被推断为常量 1。虽然模型仍能导出，但检查接口后发现输入是固定 `[1,3,224,224]`，并没有达到动态批次目标。

这说明不能只看“导出成功”字样，必须重新加载 ONNX 并检查真实接口。

### 10.2 batch=2 触发动态 `view` 转换错误

把示例输入改为 batch=2 后，Dynamo 正确保留了符号维度，但 PyTorch 2.5.1 + ONNXScript 0.1.0 在转换 ResNet 的动态 `view([batch,512])` 时触发内部错误：

```text
TypeError: unhashable type: 'list'
GraphConstructionError: Error when translating aten.view
```

这不是森林模型参数损坏，也不是输入图片错误，而是当前版本组合的新导出器在该动态形状路径上的兼容性问题。

### 10.3 最终采用的稳定方案

最终显式使用 TorchScript 导出路径：

```python
dynamo=False
dynamic_axes={
    "images": {0: "batch"},
    "logits": {0: "batch"},
}
```

结果通过了结构检查和 batch=1/2/3 的实际推理。这里不是为了使用“更新”的 API 而忽略稳定性，而是根据当前版本的真实验证结果选择可复现方案。

另外，新导出器的日志包含 Unicode 成功符号，在 Windows GBK 控制台中曾触发 `UnicodeEncodeError`。临时启用 `PYTHONUTF8=1` 后可确认这是日志编码问题，但最终使用的稳定导出路径不依赖该设置。

---

## 11. 三个代码文件分别负责什么

### 11.1 `day53_common.py`

公共逻辑包括：

- 检查并读取连续的类别索引；
- 按 50 类结构重建 ResNet18；
- 严格加载检查点；
- 定义与原项目一致的评估预处理；
- 把多张图片堆叠成 NCHW float32 batch；
- 计算文件 SHA256。

把公共逻辑单独提取出来，可以避免导出脚本和对比脚本各写一套模型结构或预处理，降低“两边代码看起来相同但实际参数不同”的风险。

### 11.2 `day53_export_onnx.py`

负责：

- 接收检查点、类别映射、ONNX 输出路径和摘要路径；
- 重建并加载模型；
- 导出动态 batch ONNX；
- 执行 `onnx.checker`；
- 执行形状推断；
- 写入模型元数据；
- 保存文件大小、SHA256、opset、节点数和环境版本。

### 11.3 `day53_compare_inference.py`

负责：

- 对真实图片执行完全相同的预处理；
- 在 PyTorch CPU 中得到参考 logits；
- 使用 ONNX Runtime CPU 得到 ONNX logits；
- 比较形状和全部数值；
- 比较 Top-1 和 Top-3；
- 把误差、图片哈希和预测结果写入 JSON 报告。

---

## 12. 完整复现步骤

以下命令在 PowerShell 中执行。

### 12.1 进入 Day53 目录

```powershell
Set-Location D:\opencv-learning\53_pytorch_onnx_export
$day53Python = 'D:\conda\envs\forest-species\python.exe'
$forestCheckpoint = 'D:\DL_code\d2l-zh\forest_species_classification\checkpoints\resnet18_baseline\best.pt'
```

### 12.2 检查依赖

```powershell
& $day53Python -m pip check
& $day53Python -c "import torch, onnx, onnxruntime, onnxscript; print(torch.__version__, onnx.__version__, onnxruntime.__version__, onnxscript.__version__)"
```

预期至少看到：

```text
No broken requirements found.
2.5.1+cu121 1.16.1 1.19.2 0.1.0
```

### 12.3 语法检查

```powershell
& $day53Python -m py_compile code\day53_common.py code\day53_export_onnx.py code\day53_compare_inference.py
```

无输出且退出码为 0 表示语法检查通过。

### 12.4 导出并校验 ONNX

```powershell
& $day53Python code\day53_export_onnx.py --checkpoint $forestCheckpoint --class-map assets\class_to_idx.json --output artifacts\forest_species_resnet18.onnx --summary outputs\export_summary.json --opset 18
```

预期末尾出现：

```text
Input: images ['batch', 3, 224, 224]
Output: logits ['batch', 50]
Classes: 50
DAY53_ONNX_EXPORT_OK
```

如果输入第一维显示为 `1` 而不是 `batch`，说明模型仍是固定批次，不能算今天的动态批次目标完成。

### 12.5 执行三图一致性比较

```powershell
& $day53Python code\day53_compare_inference.py --checkpoint $forestCheckpoint --class-map assets\class_to_idx.json --onnx-model artifacts\forest_species_resnet18.onnx --images assets\acacia_IMG_6348.JPG assets\adenanthera_microsperma_IMG_5777.JPG assets\cananga_odorata_IMG_5483.JPG --report outputs\comparison_report.json --atol 1e-4 --rtol 1e-4
```

预期末尾出现：

```text
Batch: (3, 3, 224, 224)
Output: (3, 50)
Provider: CPUExecutionProvider
DAY53_ONNX_COMPARE_OK
```

### 12.6 检查被忽略的大文件

```powershell
git check-ignore -v 53_pytorch_onnx_export\artifacts\forest_species_resnet18.onnx
git check-ignore -v 53_pytorch_onnx_export\outputs\export_summary.json
```

命令应显示它们分别命中本目录 `.gitignore` 中的 `artifacts/` 和 `outputs/` 规则。

---

## 13. 常见错误与排查顺序

### 错误 1：`class count does not match`

原因：检查点中的 `num_classes` 与 `class_to_idx.json` 数量不一致。不要强行忽略，应确认是否拿错检查点或类别映射。

### 错误 2：`Missing key` / `Unexpected key`

原因：模型结构与检查点不一致，例如最后一层类别数错误，或检查点不是标准 ResNet18。保留 `strict=True`，从源头修正结构。

### 错误 3：ONNX 只能接收 batch=1

检查 `session.get_inputs()[0].shape`。第一维应为字符串 `batch`，不是整数 1。导出时必须同时声明输入和输出的动态 batch 轴，并实际测试多个 batch 大小。

### 错误 4：ONNX 与 PyTorch 预测完全不同

优先检查预处理，而不是立刻怀疑模型：

1. RGB/BGR；
2. Resize 和 CenterCrop；
3. 0～255 是否缩放为 0～1；
4. mean/std；
5. HWC/CHW；
6. float32；
7. 模型是否 `eval()`；
8. 输入输出名称是否正确。

### 错误 5：存在小数值差异

小差异通常来自不同后端的浮点运算顺序。先用合理的 `atol/rtol` 比较全部 logits，再检查 Top-k 是否变化。不要因为不能 bit-by-bit 相等就判定导出失败，也不能只看 Top-1 相同就忽略巨大的数值偏差。

### 错误 6：以为模型会自动输出类别名

ONNX 输出只有 50 个 logits。类别名来自 `class_to_idx.json`，必须在应用层按索引映射。

### 错误 7：把 ONNX 文件直接提交 GitHub

今天的模型约 42.73 MiB，而且可由代码和原检查点重新生成。它已放入忽略目录。后续若要正式发布模型，应使用 GitHub Release、模型仓库或带校验和的下载方式，而不是随意塞进普通 Git 历史。

---

## 14. 今天没有证明什么

今天证明的是：

- 现有 PyTorch ResNet18 可以被导出为合法 ONNX；
- ONNX 接口是动态 batch 的 NCHW float32；
- 在三张同源图片上，CPU ONNX Runtime 与 CPU PyTorch 的输出数值一致；
- 该模型具备进入 C++ 推理阶段的技术前提。

今天没有证明：

- ONNX 比 PyTorch 更快；
- ONNX 模型在 GPU 上可用；
- 模型能在树莓派或 Jetson 上满足实时性；
- 模型能识别任意真实森林场景；
- 模型具备未知类别拒绝能力。

三张样本来自 BarkVN-50，与训练数据同源，只适合作为导出回归样本。实践项目 3 已经发现该模型外部 Top-1 只有 5%，因此不能把今天的三张正确预测重新包装成泛化提升。

---

## 15. 自检问题与答案

### 1. 为什么不能只保存 `state_dict` 就给 C++ 使用？

因为 `state_dict` 只有参数张量和名称，不包含一个可直接由 ONNX Runtime 执行的完整计算图。导出前仍需用正确网络结构加载参数，再转换成 ONNX 图。

### 2. 为什么导出前必须调用 `model.eval()`？

因为 BatchNorm 和 Dropout 在训练模式与推理模式下行为不同。导出推理模型时必须固定为评估行为，否则图的数值语义可能与正常部署不一致。

### 3. opset 是 ONNX Runtime 的版本吗？

不是。opset 表示 ONNX 标准算子集合的版本；ONNX Runtime 是执行引擎版本。运行时需要支持模型使用的 opset。

### 4. 动态 batch 是什么意思？

输入和输出的第 0 维在运行时可变化，例如同一个模型可接收 1、2、3 张图片。它不表示通道数和空间尺寸也全部动态。

### 5. 为什么比较 logits，而不是只看类别名称？

类别名称相同可能只是最大值索引恰好没变，无法发现其他输出已经严重漂移。比较全部 logits 可以更严格地验证数值一致性，再用 Top-k 检查实际决策是否相同。

### 6. 为什么 ONNX Runtime 使用 CPU，而 PyTorch 环境有 CUDA？

因为今天安装的是 CPU 版 `onnxruntime`。PyTorch 能使用 CUDA 与 ONNX Runtime 是否具有 CUDA Provider 是两套独立条件。CPU 对 CPU 比较也更适合作为第一步可复现基线。

### 7. C++ 端最容易出错的地方是什么？

通常不是 `session.Run()` 本身，而是 OpenCV 图片到 NCHW float32 张量的预处理，尤其是 BGR→RGB、缩放裁剪、除以 255、归一化和内存布局。

---

## 16. 今日关键记忆点

1. `.pt` 检查点不等于可跨语言部署的模型；ONNX 保存可执行计算图和权重。
2. 导出前必须重建正确模型、严格加载参数并调用 `eval()`。
3. ONNX 模型接口必须明确输入名称、dtype、NCHW 形状、动态维和输出语义。
4. 预处理通常位于 ONNX 图外，部署端必须逐项复刻。
5. `onnx.checker` 只检查结构；真正验收还要比较 PyTorch 与 ONNX Runtime 的数值和 Top-k。
6. 浮点后端允许小误差，但误差阈值必须明确，不能只凭肉眼判断。
7. 动态轴要实际用多个 batch 大小运行，不应只相信导出参数。
8. 新导出器不一定在当前版本组合下更稳定；真实错误和回退方案都应记录。
9. ONNX Runtime 的 CPU/GPU Provider 与 PyTorch 的 CUDA 状态是独立问题。
10. 模型格式转换不会修复原模型的外部泛化和未知类别拒绝问题。

---

## 17. 下一步方向

Day54 可以进入 ONNX Runtime C++ 推理的第一阶段：

1. 配置 ONNX Runtime C++ 库与 CMake；
2. 在 C++ 中读取 ONNX 输入输出名称和形状；
3. 先用可控张量运行 `Ort::Session`；
4. 再使用 OpenCV 实现与 Python 完全一致的图片预处理；
5. 比较 C++ 与今天 Python 报告中的 Top-3；
6. 明确 Windows DLL、模型文件和类别映射的部署边界。

在 C++ 端结果与 Python 基线一致之前，不进行速度宣传、模型量化或嵌入式部署。
