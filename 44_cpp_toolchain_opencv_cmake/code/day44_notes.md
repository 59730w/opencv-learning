# Day44：C++ 工具链、CMake 与 OpenCV 5

完成日期：2026-08-16
状态：已完成并通过本机构建与运行验收

## 今日目标

本课不从 C++ 基础语法重新开始，目标是把已有 C++ 与 Python/OpenCV 经验迁移到可复现的 C++ 工程：

- 使用 MSYS2 UCRT64 获得一致的 GCC、GDB、CMake、Ninja 和 OpenCV；
- 使用 `CMakePresets.json` 固定编译器、生成器和 OpenCV 查找路径；
- 用 `cv::Mat` 构造灰度矩阵并调用 `cv::threshold()`；
- 识别图像的行列、通道、深度和内存连续性；
- 理解 OpenCV 4 与 OpenCV 5 共存及迁移边界。

## 本机工具

- 编辑器：VS Code 1.108.1（x64）
- 编译器：MSYS2 UCRT64 G++ 16.2.0
- 调试器：GDB 17.2
- 构建系统：CMake 4.4.2 + Ninja 1.13.2
- C++ 库：OpenCV 5.0.0
- C++ 标准：C++17

主要路径：

```text
MSYS2_ROOT  = D:\VScode_C++\msys64
编译器      = D:\VScode_C++\msys64\ucrt64\bin\g++.exe
CMake       = D:\VScode_C++\msys64\ucrt64\bin\cmake.exe
OpenCV_DIR  = D:\VScode_C++\msys64\ucrt64\lib\cmake\opencv5
```

以上路径通过用户级环境变量配置。修改环境变量前已经打开的 VS Code 必须完全退出并重新打开，新的终端才会继承它们。

## OpenCV 4 与 OpenCV 5

- 以前 Conda 环境中的 Python OpenCV 4.10 保持不变；
- 本课的 OpenCV 5 位于独立的 MSYS2 UCRT64 目录，两者不会互相覆盖；
- `cv::Mat`、`imread`、`cvtColor`、`threshold` 等现代核心 API 可继续使用；
- OpenCV 5 最低要求 C++17，并移除了 `IplImage`、`CvMat` 等旧 C API；
- 部分高级模块被重组，复现 OpenCV 4 代码时应参考迁移说明，而不是盲目改链接库名称。

## 配置、构建与运行

在 PowerShell 中进入本课目录：

```powershell
cd D:\opencv-learning\44_cpp_toolchain_opencv_cmake
cmake --preset msys2-ucrt64
cmake --build --preset msys2-ucrt64
.\build\day44_opencv_cpp.exe
```

`CMakePresets.json` 完成三件事：

- 选择 UCRT64 GCC/G++，防止误用旧 MinGW；
- 选择 Ninja 生成器；
- 将 `OpenCV_DIR` 指向 OpenCV 5 的 CMake 配置。

程序最终应输出：

```text
OpenCV version: 5.0.0
Shape: 2 x 3
Channels: 1
Mean: 130
Depth: 0 (CV_8U = 0)
Continuous memory: true
Input pixels:
[ 10,  80, 127;
 128, 180, 255]
Binary pixels (threshold = 127):
[  0,   0,   0;
 255, 255, 255]
DAY44_OPENCV_OK
```

## `cv::Mat` 与原手写数组的对应关系

原程序用 `std::vector<int>` 表示一行像素，并手写循环完成阈值处理。本课改成：

```cpp
cv::Mat gray = ...;
cv::Mat binary;
cv::threshold(gray, binary, 127.0, 255.0, cv::THRESH_BINARY);
```

`gray` 是输入矩阵，`binary` 是输出矩阵。二者的形状均为 `2 x 3`，元素深度是 `CV_8U`，即无符号 8 位整数。

- `gray.rows == 2`，`gray.cols == 3`，`gray.channels() == 1`；
- `CV_8U` 对应单通道 `uint8` 灰度数据，取值范围是 0～255；
- `gray.ptr<std::uint8_t>()` 取得像素内存起始地址，`std::copy` 按行优先顺序写入 6 个像素；
- `gray.isContinuous() == true` 表示本例像素数据连续存储；
- `cv::mean(gray)[0]` 计算单通道平均灰度，结果为 `(10 + 80 + 127 + 128 + 180 + 255) / 6 = 130`。

## 今日实际实验

### 1. 阈值从 150 改为 127

`cv::THRESH_BINARY` 使用严格大于规则：

```text
像素值 > 127  -> 255
像素值 <= 127 -> 0
```

因此 127 保持为 0，128、180、255 变为 255，最终矩阵为：

```text
[  0,   0,   0;
 255, 255, 255]
```

只修改阈值而未更新 `expected` 时，程序按预期输出 `Unexpected threshold result.`；补充
`expected.at<std::uint8_t>(1, 0) = 255;` 后，自动验证重新通过。这说明预期矩阵能够发现处理结果的变化。

### 2. 平均灰度

使用：

```cpp
const cv::Scalar mean_value = cv::mean(gray);
std::cout << "Mean: " << mean_value[0] << '\n';
```

单通道图像读取 `cv::Scalar` 的第 0 个分量，实测输出 `Mean: 130`。

## 今日排错记录

- 第 6、7 行 OpenCV 头文件红线来自 VS Code IntelliSense 缺少 `include/opencv5`，不是 CMake 编译错误；补充包含目录后消失。
- `cmake` 一度无法识别，是因为已打开的 VS Code 终端继承了旧 PATH；为新集成终端注入 UCRT64 `bin` 后恢复。
- 用户级 PATH 使用 `D:\VScode_C++\msys64\ucrt64\bin`；系统级旧路径 `D:\VScode C++\mingw64\bin` 已删除。
- 新旧 MinGW 工具链不要在同一构建进程中混用；OpenCV C++ 项目统一使用 MSYS2 UCRT64。

## 附录：旧 MinGW 冒烟测试（非基础课程）

`day44_toolchain_smoke_test.cpp` 保留为重命名前旧 MinGW 的编译与运行证据，用于确认环境迁移没有破坏原工具链。下面的语法内容只作代码说明，不作为本阶段学习重点。

### 1. 普通变量

```cpp
int pixel_value = 10;
```

`pixel_value` 保存一个整数，这里把它看成一个灰度像素值。

### 2. 引用

```cpp
int& pixel_reference = pixel_value;
```

`pixel_reference` 是 `pixel_value` 的别名。修改引用就是修改原变量。

### 3. 指针

```cpp
int* pixel_pointer = &pixel_value;
```

- `&pixel_value` 取得变量地址；
- `pixel_pointer` 保存这个地址；
- `*pixel_pointer` 访问该地址中的值。

### 4. `std::vector`

```cpp
std::vector<int> pixels{10, 80, 127, 128, 180, 255};
```

这里用 `vector` 表示一行灰度像素。它和 Python 列表有相似之处，但其中的元素类型固定为 `int`。

### 5. 只读引用传参

```cpp
double calculate_mean(const std::vector<int>& pixels)
```

`const std::vector<int>&` 不复制整个数组，也不允许函数修改原数组，适合只读取图像数据的函数。

### 6. 可修改引用传参

```cpp
void apply_threshold(std::vector<int>& pixels, int threshold_value)
```

这里没有 `const`，函数可以直接修改原始像素数组。它模拟了 OpenCV 二值化的基本思想：

```text
像素值 >= 128  -> 255
像素值 < 128   -> 0
```

## 与之前 OpenCV 学习的联系

Python 中的 `cv2.threshold()` 与 C++ 中的 `cv::threshold()` 对应；两端的核心图像处理概念一致，主要差别是 C++ 需要显式管理类型、头文件、链接库和构建过程。

## 今日验收

以下项目均已完成：

- [x] UCRT64 G++、GDB、CMake 和 Ninja 能输出版本；
- [x] `opencv_version` 输出 `5.0.0`；
- [x] CMake 配置阶段找到 OpenCV 5；
- [x] Ninja 构建成功且无编译警告；
- [x] 能解释 Preset、CMake、Ninja、G++ 和 OpenCV 的分工；
- [x] 能解释 `cv::Mat` 的行、列、通道、深度和连续内存；
- [x] 阈值 127 实验输出第二行全 255，并通过 `expected` 自动验证；
- [x] `cv::mean(gray)[0]` 输出 `130`；
- [x] 程序输出 `DAY44_OPENCV_OK`，进程退出码为 0。

## 关键记忆点

- 编译器、OpenCV 库和运行时必须来自同一工具链；不能随意混用不同 MinGW/MSVC ABI。
- CMake 负责描述目标和依赖，Ninja 负责执行具体构建任务。
- OpenCV 5 要求 C++17；旧版 C API 和部分模块布局已经变化。
- Python OpenCV 4 与 MSYS2 OpenCV 5 位于独立环境，可以并存。
- `cv::Mat` 不只是二维数组，还携带形状、类型、通道、步长和共享数据所有权信息。
- `cv::THRESH_BINARY` 判断的是严格大于阈值；修改算法参数时，验证用的预期结果也必须同步更新。
- `cv::mean()` 返回 `cv::Scalar`；单通道图像的平均值位于第 0 个分量。
