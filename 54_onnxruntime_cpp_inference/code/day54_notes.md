# Day54：ONNX Runtime C++ 推理

## 1. 今日定位

Day53 已把森林树种 ResNet18 导出为动态批次 ONNX，并在 Python 中证明 PyTorch 与 ONNX Runtime 的 logits 和 Top-k 一致。Day54 开始把这份模型接入 C++。

今天按阶段推进：

1. 准备 ONNX Runtime C++ 依赖；
2. 建立最小 CMake 工程并检查模型接口；
3. 后续再实现 OpenCV C++ 图片预处理。

在阶段二通过前，不加入图片推理、测速、量化或嵌入式部署。

## 2. 阶段一：C++ 依赖

使用官方 CPU 版 ONNX Runtime 1.19.2，与 Day53 Python 基线保持相同版本：

```text
D:\opencv-deps\onnxruntime-win-x64-1.19.2
```

已验证：

```text
include\onnxruntime_cxx_api.h
lib\onnxruntime.lib
lib\onnxruntime.dll
```

本阶段使用 MSVC x64 19.35 和 Visual Studio CMake 3.25.1。下载压缩包在解压完整性验证后已删除，保留解压目录即可继续编译。

## 3. 阶段二：模型接口检查

### 3.1 工程文件

```text
54_onnxruntime_cpp_inference/
├── .gitignore
├── CMakeLists.txt
├── CMakePresets.json
├── code/
│   ├── day54_inspect_model.cpp
│   └── day54_notes.md
└── tests/
    └── day54_contract.ps1
```

`build/` 是生成目录，已被本日 `.gitignore` 忽略。

### 3.2 CMake 做了什么

配置时通过 `ONNXRUNTIME_ROOT` 接收外部依赖根目录，并检查头文件、导入库和 DLL 是否存在。工程把 ONNX Runtime 声明成 imported target，再让 `day54_inspect_model` 链接该目标。

Windows 动态链接包含两个时刻：

- 编译链接时使用 `onnxruntime.lib`；
- 程序启动和运行时加载 `onnxruntime.dll`。

构建完成后，CMake 使用 `copy_if_different` 把 DLL 复制到可执行文件旁边，不依赖永久修改系统 `PATH`。

### 3.3 C++ 程序做了什么

程序接收一个 ONNX 模型路径：

```text
day54_inspect_model <model.onnx>
```

然后依次完成：

1. 检查参数数量和模型文件；
2. 创建 `Ort::Env`；
3. 设置 `Ort::SessionOptions`；
4. 创建 `Ort::Session` 并加载模型；
5. 读取输入、输出数量；
6. 读取第一个输入和输出的名称、张量类型与形状；
7. 对照 Day53 森林模型契约；
8. 只有全部匹配才打印 `DAY54_MODEL_INTERFACE_OK`。

### 3.4 实际模型接口

```text
Input count: 1
Input 0 name: images
Input 0 type: float32
Input 0 shape: [batch, 3, 224, 224]
Output count: 1
Output 0 name: logits
Output 0 type: float32
Output 0 shape: [batch, 50]
```

含义：

- `batch` 是可变批次数；
- `3` 是 RGB 三通道；
- `224 × 224` 是固定输入空间尺寸；
- `50` 是 BarkVN-50 的已知类别数；
- 输入和输出都使用 `float32`；
- `logits` 是 softmax 前的原始类别分数。

## 4. 测试先行记录

阶段二先创建 PowerShell 契约测试，再创建生产代码。第一次运行测试时，可执行文件尚不存在，测试按预期失败：

```text
Executable does not exist: ...\day54_inspect_model.exe
```

这证明测试确实能阻止“没有实现却误报成功”。

最终 CTest 结果：

```text
1/1 Test #1: day54_model_interface_contract ... Passed
100% tests passed, 0 tests failed out of 1
```

## 5. 排错记录：只读视图的生命周期

第一次编译时，代码把形状辅助函数写成接收 `Ort::TensorTypeAndShapeInfo`，但 ONNX Runtime 1.19.2 从只读模型类型信息返回 `Ort::ConstTensorTypeAndShapeInfo`。修正形参类型后，程序能够编译。

第一次执行时，动态维显示为 `?`，而不是 `batch`。Python `onnx` 和 Python ONNX Runtime 都确认模型仍保存了符号维 `batch`，所以问题不在模型。

根因是：

```text
临时 Ort::TypeInfo
    ↓ 返回不拥有底层对象的 ConstTensorTypeAndShapeInfo 视图
临时对象销毁
    ↓
继续读取视图会违反生命周期要求
```

最终先保存具名 `Ort::TypeInfo`，再从它取得只读张量信息视图，并在所有读取完成后才离开函数。修正后 C++ 正确打印 `batch`。

这与之前学习的 `cv::Mat` 浅拷贝、引用和智能指针属于同一类问题：C++ 中不仅要关心“对象是什么类型”，还要关心“谁拥有资源，以及资源活多久”。

## 6. 阶段三：OpenCV C++ 预处理

### 6.1 MSVC OpenCV 依赖

原有 `OpenCV_DIR` 指向 MSYS2 UCRT64 的 GCC 版 OpenCV 5，不能与阶段二的 MSVC 程序直接混用。阶段三独立解压官方 OpenCV 4.10.0 Windows 包：

```text
D:\opencv-deps\opencv-4.10.0-msvc\opencv\build
```

验证结果：

```text
OpenCV ARCH: x64
OpenCV RUNTIME: vc16
OpenCV version: 4.10.0
```

官方包 SHA-256 验证通过，Release/Debug 的导入库和 DLL 均存在。自解压下载包在验证后已删除，现有 MSYS2 OpenCV 5 没有被修改。

### 6.2 预处理顺序

`preprocess_bgr()` 接收 OpenCV 的 `CV_8UC3` BGR 图片，并执行：

```text
BGR uint8
→ RGB uint8
→ 保持宽高比，把短边缩放到 256
→ 224 × 224 中心裁剪
→ float32，并除以 255
→ ImageNet mean/std 归一化
→ HWC 转 CHW
→ [1, 3, 224, 224] 连续 float32
```

缩小时使用 `cv::INTER_AREA`，放大时使用 `cv::INTER_LINEAR`。三个通道的归一化参数为：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### 6.3 两层测试

C++ 单元测试使用一张恒定 BGR 合成图，精确检查：

- 短边缩放与中心裁剪几何；
- BGR 转 RGB 后的通道值；
- ImageNet 归一化公式；
- CHW 内存布局；
- `[1,3,224,224]` 形状和元素数量；
- 空图片拒绝。

Python 跨语言契约测试使用 Day53 的三张真实回归图片，将 C++ 原始 float32 张量与 `day53_common.build_eval_transform()` 比较：

| 图片 | MAE | 最大绝对差 |
| --- | ---: | ---: |
| `acacia_IMG_6348.JPG` | 0.01519107 | 0.26260489 |
| `adenanthera_microsperma_IMG_5777.JPG` | 0.01601560 | 0.29761922 |
| `cananga_odorata_IMG_5483.JPG` | 0.02005792 | 0.25687128 |

验收阈值：

```text
MAE <= 0.03
最大绝对差 <= 0.35
```

这些差异来自 torchvision/PIL 抗锯齿缩放与 OpenCV `INTER_AREA` 的实现差别，不能要求逐元素为零。合成图测试负责严格排除通道顺序、归一化和布局错误，真实图测试负责限制跨图像后端的数值偏差。

阶段二和阶段三的最终 CTest 结果：

```text
day54_preprocessing_unit          Passed
day54_preprocessing_contract      Passed
day54_model_interface_contract    Passed
100% tests passed, 0 tests failed out of 3
```

成功标记：

```text
DAY54_PREPROCESS_UNIT_OK
DAY54_PREPROCESS_CONTRACT_OK
DAY54_PREPROCESS_OK
```

## 7. 复现命令

在 PowerShell 中执行：

```powershell
Set-Location D:\opencv-learning\54_onnxruntime_cpp_inference
$env:ONNXRUNTIME_ROOT = 'D:\opencv-deps\onnxruntime-win-x64-1.19.2'
$env:DAY54_OPENCV_DIR = 'D:\opencv-deps\opencv-4.10.0-msvc\opencv\build'
$env:DAY54_PYTHON = 'D:\conda\envs\forest-species\python.exe'
$vsCmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$vsCTest = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'

& $vsCmake --preset msvc-x64-release
& $vsCmake --build --preset msvc-x64-release
& $vsCTest --preset msvc-x64-release
```

检查模型接口：

```powershell
& .\build\Release\day54_inspect_model.exe `
    ..\53_pytorch_onnx_export\artifacts\forest_species_resnet18.onnx
```

生成一张图片的预处理张量：

```powershell
& .\build\Release\day54_preprocess_images.exe `
    ..\53_pytorch_onnx_export\assets\acacia_IMG_6348.JPG `
    --output .\build\acacia_tensor.bin
```

## 8. 当前阶段没有证明什么

阶段三已经证明 C++ 端完成了形状、通道、缩放裁剪、归一化和布局契约，并把与 Day53 Python 预处理的真实差异限制在明确阈值内。

阶段三尚未证明：

- C++ ONNX Runtime 可以接收该张量并执行模型；
- C++ logits 与 Day53 Python ONNX Runtime 足够接近；
- C++ Top-1/Top-3 与 Day53 基线一致；
- ONNX Runtime 比 PyTorch 更快；
- 模型外部泛化问题得到改善。

## 9. 阶段四：C++ ONNX Runtime 推理

### 9.1 最终数据流

阶段四把前面两个部分连接起来：

```text
OpenCV BGR 图片
→ 阶段三预处理
→ [batch,3,224,224] float32
→ Ort::Value
→ Ort::Session::Run()
→ [batch,50] logits
→ softmax / Top-3 / class_to_idx.json
```

`day54_cpp_inference` 支持多张图片组成动态 batch，并把输入张量与 logits 写成原始 float32 文件，供 Python 契约测试读取。程序严格验证：

- 类别索引从 0 到 49 连续且没有重复；
- 输入张量元素数量符合 batch；
- ONNX 输出是 float32；
- 输出形状为 `[batch,50]`；
- Top-k 索引能够映射回类别名称。

### 9.2 为什么分成两种一致性

最终验收不能把两个不同问题混在一起。

第一层是运行时一致性：

```text
完全相同的 OpenCV C++ 预处理张量
    ├── C++ ONNX Runtime → cpp_logits
    └── Python ONNX Runtime → python_same_tensor_logits
```

这一层排除图片解码和缩放差异，只检查 C++ 与 Python ONNX Runtime 是否执行了同一个模型计算。验收使用 `atol=1e-4`、`rtol=1e-4`，并要求有序 Top-3 完全一致。

第二层是端到端一致性：

```text
同一张 JPEG
    ├── OpenCV C++ INTER_AREA → C++ ONNX Runtime
    └── torchvision/PIL antialias → Python ONNX Runtime
```

这一层同时包含图像后端差异。阶段三已经证明两个张量接近但不逐元素相同，因此这里要求 Top-1 相同、Top-3 类别集合相同，并如实记录排序是否变化。端到端 logits 只用于量化后端差异，阈值为 MAE 不超过 0.30、最大绝对差不超过 1.0。

### 9.3 batch=1 和 batch=3 数值结果

| batch | 同张量 logits MAE | 同张量最大差 | 端到端 logits MAE | 端到端最大差 | Top-1 | Top-3 集合 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.00000000 | 0.00000000 | 0.19381276 | 0.51231194 | 相同 | 相同 |
| 3 | 0.00000000 | 0.00000000 | 0.20271154 | 0.86522293 | 3/3 相同 | 3/3 相同 |

同一张量经过 C++ 和 Python ONNX Runtime 时，当前 CPU 环境下 logits 逐元素相同，有序 Top-3 也完全一致。这证明 C++ 的 `Ort::Session::Run()` 接入本身没有引入数值漂移。

与 Day53 PIL 基线相比，三张图的 Top-1 和 Top-3 类别集合全部相同，但 Acacia 的第 2、3 名顺序互换：

```text
Day53 PIL：     Acacia → Annona squamosa → Prunus salicina
Day54 OpenCV：  Acacia → Prunus salicina → Annona squamosa
```

因此端到端有序 Top-3 是 2/3 完全一致，而不是 3/3。该差异来自预处理插值，不能写成 ONNX Runtime 推理错误，也不能隐藏成“全部 Top-3 完全一致”。

### 9.4 C++ 实际 batch=3 输出

```text
Batch size: 3
Input shape: [3, 3, 224, 224]
Output shape: [3, 50]

Acacia:
1. Acacia
2. Prunus salicina
3. Annona squamosa

Adenanthera microsperma:
1. Adenanthera microsperma
2. Magnolia alba
3. Polyalthia longifolia

Cananga odorata:
1. Cananga odorata
2. Tamarindus indica
3. Cedrus
```

batch=3 原始文件大小也通过检查：

```text
输入张量：3 × 3 × 224 × 224 × 4 = 1,806,336 bytes
输出 logits：3 × 50 × 4 = 600 bytes
```

确定性成功标记：

```text
DAY54_CPP_INFERENCE_OK
DAY54_INFERENCE_CONTRACT_OK
```

### 9.5 最终测试

```text
day54_preprocessing_unit          Passed
day54_preprocessing_contract      Passed
day54_model_interface_contract    Passed
day54_inference_contract          Passed
100% tests passed, 0 tests failed out of 4
```

阶段四是 Day54 的最后一个功能阶段。本地功能与验证已经完成，当前等待学习者检查本笔记；在学习者确认之前，不提交、不推送 GitHub。

## 10. 阶段四复现命令

```powershell
Set-Location D:\opencv-learning\54_onnxruntime_cpp_inference
New-Item -ItemType Directory -Force -Path outputs | Out-Null

& .\build\Release\day54_cpp_inference.exe `
    --model ..\53_pytorch_onnx_export\artifacts\forest_species_resnet18.onnx `
    --class-map ..\53_pytorch_onnx_export\assets\class_to_idx.json `
    --tensor-output .\outputs\batch3_tensor.bin `
    --logits-output .\outputs\batch3_logits.bin `
    --images `
        ..\53_pytorch_onnx_export\assets\acacia_IMG_6348.JPG `
        ..\53_pytorch_onnx_export\assets\adenanthera_microsperma_IMG_5777.JPG `
        ..\53_pytorch_onnx_export\assets\cananga_odorata_IMG_5483.JPG
```

`outputs/` 已被忽略，不上传原始张量和 logits。

## 11. Day54 证明了什么

Day54 证明：

- 官方 ONNX Runtime 1.19.2 和 OpenCV 4.10.0 可以通过 MSVC/CMake 接入；
- C++ 能读取模型的动态 batch、输入输出名称、shape 和 dtype；
- C++ 能复现完整的 OpenCV 图片预处理契约；
- 同一输入张量在 C++ 与 Python ONNX Runtime CPU 中得到一致 logits；
- batch=1 和 batch=3 均能实际运行；
- C++ 能读取 50 类映射并输出 Top-1/Top-3；
- PIL 与 OpenCV 的插值差异能够被量化并影响低排名类别顺序。

Day54 没有证明：

- C++ 推理比 Python 或 PyTorch 更快；
- GPU、TensorRT、量化或嵌入式部署已经可用；
- 模型外部 Top-1 从实践项目 3 的 5% 得到改善；
- 闭集 softmax 可以拒绝未知树种；
- 三张同源回归图代表真实森林泛化能力。

## 12. 后续课程素材策略

以后适合多样性测试的课程，不再默认反复使用单张图片。优先从已核验图片池中选择约 4～8 张可审查的小型代表集放入当天 `assets/`，覆盖尺寸、方向、内容、来源和难度差异，并在笔记中记录原始来源与用途。仍然不把整个数据集批量复制进学习仓库。

Day54 为保证与 Day53 已冻结回归基线可比，继续使用现有三张同源图片，没有在最终门临时扩大或改变测试集。

## 13. 关键记忆点

1. ONNX 模型加载成功不等于模型接口符合预期，必须读取并检查真实接口。
2. Windows 链接时使用 `.lib`，运行时需要 `.dll`，不同编译器的库不能随意混用。
3. 动态批次必须通过多个实际 batch 运行，不能只看符号维声明。
4. `ConstTensorTypeAndShapeInfo` 是不拥有资源的视图，其所有者必须保持存活。
5. OpenCV 默认 BGR，而模型训练契约是 RGB；转换顺序不能省略。
6. HWC 转 CHW 是内存布局转换，不只是打印一个不同的形状。
7. 应把同张量运行时一致性与不同图像后端的端到端一致性分开验证。
8. Top-1、Top-3 集合和有序 Top-3 是三种不同强度的结论，不能混写。
9. 模型格式转换和 C++ 部署不会修复原模型的外部泛化或未知类别拒绝问题。
