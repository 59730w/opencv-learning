# Day46：C++17 文件系统与批量图像处理

## 学习定位

Day1～Day15 已经用 OpenCV-Python 学过图像读取、滤波、颜色等基础算子，Day44/45 完成了 C++ 工具链、CMake、OpenCV 5 环境和 `cv::Mat` 内存语义。Day46 不重复这些基础算子，而是把重点放在 C++ 特有的能力上：

- 用 C++17 `std::filesystem` 遍历目录、按扩展名筛选图片；
- 用可复用函数边界组织程序（每个函数只做一件事）；
- 逐张 `cv::imread` 批量读取，单文件损坏只跳过、不终止整个批量（失败隔离）；
- 用 `struct` + `std::vector` 汇总每张图的结果并打印表格；
- 记录 OpenCV 4 → OpenCV 5 的类型编码变化。

## 环境

- 编译器：MSYS2 UCRT64 G++ 16.2.0
- C++ 标准：C++17
- 构建工具：CMake 4.4.2 + Ninja 1.13.2
- OpenCV：5.0.0（本次只链接 `core` 与 `imgcodecs`，不需要 `highgui`）
- MSYS2 根目录：`D:\VScode_C++\msys64`

## 目录结构

```text
46_cpp_batch_image_processor/
├── assets/
│   ├── day45_input.jpg    # 正常测试图片（427 x 640）
│   └── broken.jpg         # 故意损坏的测试样本（纯文本内容，非图片）
├── code/
│   ├── day46_batch_processor.cpp
│   └── day46_notes.md
├── .gitignore
├── CMakeLists.txt
└── CMakePresets.json
```

`build/`、exe 和目标文件由 `.gitignore` 排除，不上传 GitHub。

## 1. `std::filesystem` 目录遍历

```cpp
namespace fs = std::filesystem;

std::vector<fs::path> list_images(const fs::path& directory) {
    std::vector<fs::path> images;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && is_image_extension(entry.path())) {
            images.push_back(entry.path());
        }
    }
    return images;
}
```

- `fs::directory_iterator(dir)` 遍历目录下的条目，**不会递归进入子目录**；
- `entry.is_regular_file()` 判断是否为普通文件（排除子目录、快捷方式等）；
- `entry.path()` 得到 `fs::path`，`.extension()` 取扩展名（如 `.jpg`）；
- `directory_iterator` 的遍历顺序**不保证**：本次实测 `broken.jpg` 有时排在 `day45_input.jpg` 前面，程序逻辑不能依赖文件顺序。

## 2. 函数边界与退出码

程序拆成 4 个可复用函数：

| 函数 | 职责 |
|---|---|
| `is_image_extension(path)` | 判断扩展名是否属于图片 |
| `list_images(directory)` | 收集目录下所有图片路径 |
| `load_image(path)` | 读取一张图并打包成 `ImageResult` |
| `print_summary(results)` | 打印结果汇总表 |

主函数只负责编排流程：检查参数 → 收集路径 → 逐张加载 → 汇总。每个函数只做一件事，方便单独测试和复用。

退出码约定：

| 情况 | 退出码 |
|---|---|
| 批量处理正常完成（含跳过损坏文件） | 0 |
| 目录不存在或不是目录 | 1 |
| 没有提供唯一目录参数 | 2 |

## 3. OpenCV 批量读取与失败隔离（核心）

```cpp
ImageResult load_image(const fs::path& path) {
    ImageResult result;
    result.path = path;
    const cv::Mat mat = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (!mat.empty()) {
        result.loaded = true;
        result.width = mat.cols;
        result.height = mat.rows;
    }
    return result;
}
```

- 读取失败时 `cv::imread` 返回空 `cv::Mat`，用 `mat.empty()` 判断（对应 Python 的 `image is None`）；
- **失败隔离**：`empty()` 时只把该文件记为失败（`loaded = false`），主循环继续处理下一张，**整批不终止**；
- 这是批处理程序的必备能力：一张损坏图片不能拖垮整个任务。

测试样本 `assets\broken.jpg` 的内容是一行纯文本，扩展名是 `.jpg`，会被 `list_images` 收集，但 `imread` 读不出任何像素，正好用来验证失败隔离。

## 4. `struct` + `std::vector` 结果汇总

```cpp
struct ImageResult {
    fs::path path;
    bool loaded = false;
    int width = 0;
    int height = 0;
};

std::vector<ImageResult> results;
results.reserve(images.size());
for (const fs::path& image : images) {
    results.push_back(load_image(image));
}
print_summary(results);
```

- 每个 `ImageResult` 记录一张图的结果（路径、是否成功、宽、高）；默认成员初始化保证未设置字段有合理初值；
- `results.reserve(n)` 提前预留容量，避免 `push_back` 过程中反复扩容拷贝；
- `print_summary` 用 `const std::vector<ImageResult>&` 传参：不复制整个容器，也不允许函数修改（Day45 的只读引用知识点直接复用）。

## 5. OpenCV 4 → OpenCV 5 类型编码变化

本次打印 `mat.type()` 得到 **64**，而 OpenCV 4 里同一张 `CV_8UC3` 图是 **16**。用预处理器宏展开核实：

```text
#define CV_8U 0
#define CV_8UC3 CV_MAKETYPE(CV_8U, 3)
#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_CN_SHIFT 5
```

OpenCV 5 把 `CV_CN_SHIFT` 从 3 改为 5（通道位域加宽，可支持更多通道数）：

```text
CV_8UC3 = 0 + (3-1) << 5 = 64    （OpenCV 5）
CV_8UC3 = 0 + (3-1) << 3 = 16    （OpenCV 4）
```

**经验**：跨版本比较类型用宏（`type() == CV_8UC3`），不要用数值（`type() == 16`）。

## 6. CMake：本次只需要 `core` + `imgcodecs`

```cmake
find_package(OpenCV 5 REQUIRED COMPONENTS core imgcodecs)
target_link_libraries(day46_batch_processor PRIVATE
    opencv_core
    opencv_imgcodecs
)
```

- 批量处理器不弹窗显示，**不需要链接 `highgui`**（对比 Day45 链接了 `core + imgcodecs + highgui`）；
- 阶段 A 的 CMakeLists 还没写 `find_package` 时，CMake 警告 `OpenCV_DIR` 未被使用——`CMakePresets.json` 里的 `OpenCV_DIR` 只有 CMakeLists 真正调用 `find_package(OpenCV)` 才会生效；阶段 B 补上后警告消失。这解释了"Preset 里写了变量 ≠ 自动生效"。

## 构建与运行

```powershell
cd D:\opencv-learning\46_cpp_batch_image_processor
$env:MSYS2_ROOT = 'D:\VScode_C++\msys64'
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"
cmake --preset msys2-ucrt64
cmake --build --preset msys2-ucrt64
.\build\day46_batch_processor.exe .\assets
```

## 最终验收记录

成功路径（`assets` 目录含 1 张正常图 + 1 张损坏图；`broken.jpg` 本次排在前面，证明顺序不依赖）：

```text
Found 2 image(s):
Result summary:
  SKIP: .\assets\broken.jpg (unreadable)
  OK:   .\assets\day45_input.jpg  427 x 640
Loaded 1 / 2 image(s); skipped 1 unreadable file(s).
DAY46_BATCH_OK
```

退出码通过 `$LASTEXITCODE` 确认：

```text
成功路径退出码    = 0
无参数退出码      = 2
目录不存在退出码  = 1
```

## 关键记忆点

1. `std::filesystem::directory_iterator` 只遍历当前目录、不递归，遍历顺序不保证，逻辑不能依赖顺序。
2. 读图失败用 `image.empty()` 判断；批量处理必须做单文件失败隔离——失败只记录，不终止整批。
3. `struct` 把一张图的多个结果字段打包，`std::vector` 存多张图的结果；`reserve` 提前预留容量减少拷贝。
4. 只读容器参数用 `const std::vector<T>&`：不复制、不改动。
5. 函数边界：每个函数只做一件事，主函数只编排流程。
6. OpenCV 5 的 `CV_CN_SHIFT` 从 3 变为 5，`CV_8UC3` 的数值从 16 变为 64；跨版本比较类型用宏不用数值。
7. CMake 链接哪些模块取决于程序真正用到什么：批量处理不显示窗口就不需要 `highgui`。

## Day47 计划

Day46 已具备"批量读 + 汇总"。Day47 的自然延伸（任选其一，以实际安排为准）：

- 批量处理与落盘：每张图转灰度（`cv::cvtColor`）后 `cv::imwrite` 到输出目录，练习 `fs::create_directory` 与输出路径管理；
- 递归遍历：`recursive_directory_iterator` 处理嵌套目录；
- 调试与部署：GDB 调试、运行时 DLL 依赖（`ldd` / `objdump -p`）分析。
