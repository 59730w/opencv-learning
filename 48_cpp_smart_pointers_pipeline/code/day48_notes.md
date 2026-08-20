# Day48：智能指针与可配置图像处理管线

## 学习定位

Day47 已经完成“遍历目录 → 读取图片 → 灰度处理 → 保存结果 → 失败隔离”。Day48 在此基础上完成两件事：

1. 使用小型程序理解 `std::unique_ptr`、`std::shared_ptr` 和 RAII 所有权；
2. 把固定灰度处理升级为由 `--op` 参数选择 `gray`、`blur` 或 `edge` 的可配置批处理器。

本日没有为了展示智能指针而使用 `std::unique_ptr<cv::Mat>`。`cv::Mat` 已经具有自动生命周期管理和底层数据引用计数，再套一层智能指针通常只会增加间接访问和所有权理解成本。

## 环境

- 编译器：MSYS2 UCRT64 G++ 16.2.0
- C++ 标准：C++17
- 构建工具：CMake 4.4.2 + Ninja 1.13.2
- OpenCV：5.0.0（`core`、`imgcodecs`、`imgproc`）
- 警告选项：`-Wall -Wextra -pedantic`
- MSYS2 根目录：`D:\VScode_C++\msys64`

## 目录结构

```text
48_cpp_smart_pointers_pipeline/
├── assets/
│   ├── broken.jpg
│   └── day45_input.jpg
├── code/
│   ├── day48_configurable_pipeline.cpp
│   ├── day48_notes.md
│   └── day48_smart_pointers.cpp
├── tests/
│   └── day48_contract.ps1
├── .gitignore
├── CMakeLists.txt
└── CMakePresets.json
```

`build/` 和 `output/` 是本地生成目录，由 `.gitignore` 排除。

## 1. RAII 与所有权

RAII（Resource Acquisition Is Initialization）把资源的生命周期绑定到对象生命周期：对象构造时获得资源，对象离开作用域时由析构函数自动释放资源。智能指针是 C++ 标准库中管理动态对象所有权的 RAII 工具。

“所有权”回答的是：谁负责保证资源最终被释放一次。裸指针只能表示地址，本身没有说明谁负责释放；智能指针把这项责任写进类型。

## 2. `std::unique_ptr` 与 `std::move`

`unique_ptr` 表示独占所有权，不能复制，只能移动：

```cpp
auto owner = std::make_unique<TrackedResource>("camera-buffer");
std::unique_ptr<TrackedResource> next_owner = std::move(owner);
```

- 禁止复制是为了避免两个独占所有者重复释放同一资源；
- `std::move` 本身不搬动资源，它把表达式转换为可以触发移动操作的形式；
- 移动构造完成后，资源由 `next_owner` 管理，原来的 `owner == nullptr`；
- 默认需要动态所有权时，应优先考虑 `unique_ptr`。

## 3. `std::shared_ptr` 与引用计数

`shared_ptr` 用于多个所有者确实需要共同维持对象生命周期的情况：

```cpp
auto first = std::make_shared<TrackedResource>("shared-config");
{
    std::shared_ptr<TrackedResource> second = first;
}
```

- `use_count()` 表示当前参与所有权的 `shared_ptr` 数量；
- 复制 `shared_ptr` 后计数增加，所有者销毁或 `reset()` 后计数减少；
- 最后一个所有者消失、计数变为 0 时，对象才被释放；
- `weak_ptr` 不增加引用计数，可用于观察共享对象和打破循环引用，本日不展开编码。

本日实测计数变化为 `1 → 2 → 1`，两个演示资源都只析构一次。

## 4. 为什么 `cv::Mat` 不需要再套智能指针

`cv::Mat` 是一个具有 RAII 行为的轻量矩阵头。复制 `cv::Mat` 时通常共享底层图像数据并增加引用计数；最后一个相关 `cv::Mat` 销毁时，底层数据自动释放。

```cpp
cv::Mat image2 = image1;          // 浅拷贝，共享底层数据
cv::Mat image3 = image1.clone();  // 深拷贝，独立底层数据
```

常见接口按值返回 `cv::Mat`，只读输入使用 `const cv::Mat&`。写成 `std::unique_ptr<cv::Mat>` 通常不会改善底层像素所有权，反而增加一次指针间接访问。

## 5. `enum class` 与 `--op` 参数解析

命令格式为：

```text
day48_pipeline <input_dir> <output_dir> --op <gray|blur|edge>
```

用户输入首先由 `parse_operation` 转成强类型枚举：

```cpp
enum class Operation { Gray, Blur, Edge };
```

这样只有程序边界处理字符串，内部函数统一接收 `Operation`，避免到处比较裸字符串。无法识别的操作返回 `std::nullopt`，程序打印错误并返回退出码 4。

## 6. `gray` / `blur` / `edge` 三种处理操作

| 操作 | OpenCV 调用 | 输出 | 实测类型 |
| --- | --- | --- | ---: |
| `gray` | `cvtColor(..., COLOR_BGR2GRAY)` | 灰度图 | `CV_8UC1 = 0` |
| `blur` | `GaussianBlur(..., Size(7, 7), 1.5)` | 三通道平滑图 | `CV_8UC3 = 64` |
| `edge` | 先灰度化，再 `Canny(..., 80, 160)` | 单通道二值边缘响应 | `CV_8UC1 = 0` |

输出文件名由输入主文件名和操作名组成，例如：

```text
day45_input.jpg -> day45_input_gray.jpg
day45_input.jpg -> day45_input_blur.jpg
day45_input.jpg -> day45_input_edge.jpg
```

目视检查结果：灰度图颜色被正确移除；模糊图细节明显变平滑；边缘图以黑色背景和白色轮廓呈现树木、地平线及地面纹理。

## 7. 失败隔离与退出码

| 场景 | 退出码 | 行为 |
| --- | ---: | --- |
| 整批正常结束，允许单图读写失败 | 0 | 记录结果并继续 |
| 输入路径不存在或不是目录 | 1 | 致命输入错误 |
| 参数数量或 `--op` 位置错误 | 2 | 打印 Usage |
| 输出目录创建失败 | 3 | 致命输出错误 |
| 未知操作名 | 4 | 打印 Unknown operation |

`broken.jpg` 是故意损坏的文件。三种操作都将它记录为 `SKIP (read-failed)`，同时继续处理正常图片。

## 构建与运行

```powershell
cd D:\opencv-learning\48_cpp_smart_pointers_pipeline
$env:MSYS2_ROOT = 'D:\VScode_C++\msys64'
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"

cmake --preset msys2-ucrt64
cmake --build --preset msys2-ucrt64 --clean-first

.\build\day48_smart_pointers.exe
.\build\day48_pipeline.exe .\assets .\output\gray --op gray
.\build\day48_pipeline.exe .\assets .\output\blur --op blur
.\build\day48_pipeline.exe .\assets .\output\edge --op edge
```

完整自动验收：

```powershell
.\tests\day48_contract.ps1
```

## 最终验收记录

智能指针程序：

```text
[unique_ptr]
construct: camera-buffer
source empty: true
new owner: camera-buffer
destroy: camera-buffer
[shared_ptr]
construct: shared-config
count: 1
count: 2
second sees: shared-config
count: 1
destroy: shared-config
DAY48_SMART_POINTERS_OK
EXIT=0
```

三种处理操作：

```text
gray: Saved 1 / 2; read-failed 1; write-failed 0; type=0; exit=0
blur: Saved 1 / 2; read-failed 1; write-failed 0; type=64; exit=0
edge: Saved 1 / 2; read-failed 1; write-failed 0; type=0; exit=0
```

自动契约测试共通过 25 项检查，最终输出：

```text
DAY48_CONTRACT_OK
TEST_EXIT=0
```

负路径实测：

```text
未知操作 sharpen：退出码 4
缺少参数：退出码 2
输入目录不存在：退出码 1
输出路径被普通文件占用：退出码 3
```

## 遇到的问题与解决方法

第一次只构建智能指针 target 时，preset 中的 `OpenCV_DIR` 尚未被使用，CMake 提示 `unused-cli`。加入依赖 OpenCV 的 `day48_pipeline` target 后重新配置，并执行 clean build，警告消失；两个 targets 均成功编译。

测试采用 RED-GREEN 流程：先创建契约测试，确认它因为两个可执行文件不存在而失败；完成智能指针 target 后只剩管线 target 缺失；完成管线后 25 项检查全部通过。

## 关键记忆点

1. `unique_ptr` 表示独占所有权，不能复制，只能移动。
2. `std::move` 允许触发所有权转移；移动完成后源 `unique_ptr` 为空。
3. `shared_ptr` 通过控制块记录共享所有者，最后一个所有者消失时释放对象。
4. 默认优先 `unique_ptr`；只有真实共享生命周期时才使用 `shared_ptr`。
5. `cv::Mat` 已具有 RAII 和引用计数，通常按值或 `const cv::Mat&` 传递。
6. 把字符串参数在边界转换为 `enum class`，内部处理逻辑更清晰、更安全。
7. 可配置管线仍需保留 Day47 的单文件读写失败隔离和确定性退出码。

## Day49 计划

学习 GDB 调试和 Windows 运行时 DLL 分析：使用断点、单步、变量观察和调用栈定位问题，并理解程序在终端可运行但脱离工具链环境可能缺少 DLL 的原因。
