# Day45：OpenCV C++ 图像 I/O、`cv::Mat` 内存语义与工程验收

## 学习定位

Day1～Day15 已经使用 OpenCV-Python 学过图像读取、颜色、像素、绘图、变换、视频、直方图和滤波。Day45 不重复这些基础算子，而是把重点放在 C++ 特有的类型、内存、函数接口、CMake 链接和错误处理上。

## 环境

- 编译器：MSYS2 UCRT64 G++ 16.2.0
- C++ 标准：C++17
- 构建工具：CMake 4.4.2 + Ninja 1.13.2
- OpenCV：5.0.0
- HighGUI 运行时：Qt6 Base 6.11.1 + Qt6 5Compat 6.11.1
- MSYS2 根目录：`D:\VScode_C++\msys64`

## 目录结构

```text
45_opencv_cpp_image_io/
├── assets/
│   └── day45_input.jpg
├── code/
│   ├── day45_image_io.cpp
│   └── day45_notes.md
├── .gitignore
├── CMakeLists.txt
└── CMakePresets.json
```

`build/`、exe 和目标文件由 `.gitignore` 排除，不上传 GitHub。

## 1. 从文件读取 `cv::Mat`

```cpp
const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

if (image.empty()) {
    std::cerr << "Could not read image: " << image_path.string() << '\n';
    return 1;
}
```

Python 中通常用 `image is None` 判断读取失败；C++ 中用 `image.empty()`。正常彩色 JPG 默认按 BGR 读取，通常得到 `CV_8UC3`。

本次图片属性：

- 宽度：427，即 `cols`
- 高度：640，即 `rows`
- 通道数：3
- 深度：`CV_8U`
- 类型：`CV_8UC3`
- 内存连续：`true`

## 2. HighGUI 显示与构建依赖

```cpp
cv::imshow("Day45 Input", image);
cv::waitKey(0);
cv::destroyAllWindows();
```

- `imshow` 创建窗口并显示 `cv::Mat`。
- `waitKey(0)` 处理窗口事件并等待键盘输入。
- `destroyAllWindows()` 关闭 OpenCV 窗口。

使用一个 OpenCV 模块需要同时检查三层：

1. C++ 是否包含对应头文件，例如 `opencv2/highgui.hpp`。
2. CMake 是否查找并链接对应模块，例如 `highgui` 和 `opencv_highgui`。
3. 运行时依赖 DLL 是否安装，并能通过 `PATH` 找到。

本次 `undefined reference to cv::imshow`、`cv::waitKey` 和 `cv::destroyAllWindows` 的直接原因是 CMake 没有链接 `opencv_highgui`。补上链接后又发现 HighGUI 所需 Qt6 DLL 未安装，最终补装：

```text
mingw-w64-ucrt-x86_64-qt6-base
mingw-w64-ucrt-x86_64-qt6-5compat
```

退出时出现的 `QThreadStorage: entry ... destroyed before end of thread` 位于成功标志之后；本次程序仍以退出码 0 正常完成，因此把它记录为当前 MSYS2 Qt6 后端的非阻塞退出提示。

## 3. `cv::Mat` 浅拷贝与深拷贝

```cpp
const cv::Mat shallow_copy = image;
const cv::Mat deep_copy = image.clone();
```

- 普通赋值只复制 `cv::Mat` 矩阵头，两个对象共享同一像素缓冲区。
- 浅拷贝是 O(1) 操作，会增加引用计数，但不会复制全部像素。
- `clone()` 分配新缓冲区并复制像素，得到真正独立的数据。
- 本次实测 `image.data == shallow_copy.data`，而 `image.data != deep_copy.data`。

## 4. `const cv::Mat&` 函数接口

```cpp
bool shares_pixel_buffer(const cv::Mat& first, const cv::Mat& second) {
    return first.data == second.data;
}
```

按值传递 `cv::Mat` 不会深拷贝像素，但会复制矩阵头并增减引用计数。使用 `const cv::Mat&` 可以直接绑定现有对象，避免这次矩阵头复制，并表达正常情况下只读取输入的接口意图。

## 5. 命令行参数与退出码

程序约定：

| 情况 | 退出码 | 含义 |
|---|---:|---|
| 图片读取、检查和显示成功 | 0 | 正常完成 |
| 图片路径不存在或无法读取 | 1 | 任务执行失败 |
| 没有提供唯一图片参数 | 2 | 调用方式错误 |
| `cv::Mat` 拷贝行为与预期不符 | 3 | 内部验收失败 |

PowerShell 必须在 exe 结束后立刻读取 `$LASTEXITCODE`。

## 构建与运行

```powershell
cd D:\opencv-learning\45_opencv_cpp_image_io
$env:MSYS2_ROOT = 'D:\VScode_C++\msys64'
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"
cmake --preset msys2-ucrt64
cmake --build --preset msys2-ucrt64
.\build\day45_image_io.exe .\assets\day45_input.jpg
```

## 最终验收记录

成功路径已确认：

```text
Shallow copy shares data: true
Deep copy owns different data: true
DAY45_MAT_OWNERSHIP_OK
DAY45_CONST_REFERENCE_OK
DAY45_IMAGE_DISPLAY_OK
VALID_IMAGE_EXIT_CODE=0
```

失败路径已确认：

```text
NO_ARGUMENT_EXIT_CODE=2
MISSING_IMAGE_EXIT_CODE=1
```

## 关键记忆点

1. `rows` 是高度，`cols` 是宽度。
2. `depth()` 只表示单通道数据深度，`type()` 同时包含深度和通道数。
3. 头文件让编译器认识声明，链接库提供函数实现，运行时 DLL 让程序真正启动。
4. `cv::Mat` 普通赋值共享像素数据；需要独立副本时使用 `clone()` 或 `copyTo()`。
5. 只读函数参数优先考虑 `const cv::Mat&`。
6. 成功标志和退出码共同构成可重复的命令行验收证据。

## Day46 计划

不重复基础 OpenCV 算子。使用 C++17 `std::filesystem`、STL 容器和可复用函数，构建一个能够遍历目录、逐张读取图片、隔离单文件错误并汇总结果的批处理小程序。
