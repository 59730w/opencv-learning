# Day47：批量图像处理与结果落盘

## 学习定位

Day46 已实现"遍历目录 → 批量读取 → 失败隔离 → 汇总"。Day47 在它之上加了**处理与输出**：把每张图片转成灰度并保存到输出目录。新知识点集中在 C++/OpenCV 工程细节：

- `std::filesystem::create_directories` 创建输出目录（幂等、可建多级、不抛异常）；
- `cv::cvtColor` 颜色空间转换（灰度化，`imgproc` 模块）；
- `cv::imwrite` 保存图片——**返回 `bool`，写失败也要检查**；
- `fs::path` 的 `operator/` 与 `stem()` 构造输出路径；
- 失败隔离从"读"扩展为"读/写"两个维度。

## 环境

- 编译器：MSYS2 UCRT64 G++ 16.2.0
- C++ 标准：C++17
- 构建工具：CMake 4.4.2 + Ninja 1.13.2
- OpenCV：5.0.0（链接 `core` + `imgcodecs` + `imgproc`）
- MSYS2 根目录：`D:\VScode_C++\msys64`

## 目录结构

```text
47_cpp_batch_image_processing/
├── assets/
│   ├── day45_input.jpg    # 正常测试图片（427 x 640）
│   └── broken.jpg         # 故意损坏的测试样本（纯文本内容，非图片）
├── output/                # 运行产物（灰度图），被 .gitignore 排除
├── code/
│   ├── day47_batch_processing.cpp
│   └── day47_notes.md
├── .gitignore             # 新增 output/
├── CMakeLists.txt         # 新增 imgproc 模块
└── CMakePresets.json
```

`build/`、`output/`、exe 和目标文件由 `.gitignore` 排除，不上传 GitHub。

## 1. 双参数与 `std::error_code`

程序现在接收两个参数：输入目录、输出目录。

```cpp
if (argc != 3) {
    std::cerr << "Usage: day47_batch_processing <input_dir> <output_dir>\n";
    return 2;
}

std::error_code ec;
fs::create_directories(output_dir, ec);
if (ec) {
    std::cerr << "Cannot create output directory: " << output_dir.string()
              << " (" << ec.message() << ")\n";
    return 3;
}
```

- `argc != 3`：两个参数，比 Day46 多一个；
- `std::error_code`：`create_directories` 的**不抛异常重载**，错误放进 `ec` 而不是让程序崩溃；
- 新增退出码 `3`（输出目录创建失败），退出码约定变为：

| 情况 | 退出码 |
|---|---|
| 批量处理正常完成（含跳过/写失败） | 0 |
| 输入目录不存在或不是目录 | 1 |
| 参数个数不对 | 2 |
| 输出目录创建失败 | 3 |

## 2. `create_directories`：幂等、可建多级

- `fs::create_directory` 只能创建一级目录；`fs::create_directories` 能一次创建多级（如 `a/b/c`）；
- **幂等**：目录已存在时返回 `false` 但 `ec` 无错误——重复运行不会报错（实测连续运行两次均退出码 0）；
- 失败原因（如路径被同名普通文件占用）会写入 `ec`，可用 `ec.message()` 查看。

## 3. `cv::cvtColor` 与 `cv::imwrite`：处理与落盘

```cpp
cv::Mat gray;
cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

const fs::path out = grayscale_output_path(image, output_dir);
if (!cv::imwrite(out.string(), gray)) {
    ++write_failed;
    std::cout << "  FAIL (write-failed): " << out.string() << '\n';
    continue;
}
```

- `cvtColor(src, dst, code)`：颜色空间转换；OpenCV 读图默认 BGR，`COLOR_BGR2GRAY` 得到单通道灰度图；
- **`cvtColor` 属于 `imgproc` 模块**——CMake 的 `find_package` 与 `target_link_libraries` 都要加 `imgproc`；
- **`cv::imwrite` 返回 `bool`**：写失败返回 `false`（OpenCV 同时会打 WARN 日志，例如 `can't open file for writing`），**程序判断靠返回值，不能只看日志**；
- 灰度图 `gray.type()` 实测为 `0`，即 `CV_8UC1`（对照 Day46：OpenCV 5 中 `CV_8UC1 = 0 + (1-1)<<5 = 0`，彩色 `CV_8UC3 = 64`）。

## 4. `fs::path` 运算：构造输出路径

```cpp
fs::path grayscale_output_path(const fs::path& image_path,
                               const fs::path& output_dir) {
    return output_dir / (image_path.stem().string() + "_gray.jpg");
}
```

- `operator/`：路径拼接（`输出目录 / 文件名`）；
- `stem()`：取主文件名（不含扩展名），`day45_input.jpg` → `day45_input`；
- 输出命名：`day45_input.jpg` → `day45_input_gray.jpg`；
- **工程坑**：若输入目录同时有 `a.jpg` 和 `a.png`，两者都会生成 `a_gray.jpg`，后写者覆盖先写者——真实批量任务要注意文件名冲突。

## 5. 读/写双重失败隔离

Day46 只有"读失败"一个维度；Day47 增加"写失败"：

- **读失败**：`imread` 返回空 `Mat`（`empty()`）→ `SKIP (read-failed)`，继续下一张；
- **写失败**：`imwrite` 返回 `false` → `FAIL (write-failed)`，继续下一张；
- 任何一张图的问题都不终止整批；只有输出目录创建失败（退出码 3）才是致命错误。

写失败实测方法：在输出目录里放一个与目标文件**同名**的目录（`out2\day45_input_gray.jpg`），`imwrite` 无法打开该路径 → 返回 `false` → 程序打印 `FAIL (write-failed)` 后继续，整批仍以退出码 0 完成。

## 6. `struct ImageResult`：结果汇总表（阶段 C）

```cpp
struct ImageResult {
    fs::path path;
    bool read_ok = false;
    bool saved = false;
    int width = 0;
    int height = 0;
    int type = 0;
    fs::path output_path;
};
```

- 每张图一个 `ImageResult`，`std::vector` 收集（`reserve` 预留容量）；
- `process_image(path, output_dir)` 封装"读 → 转 → 写"完整流程并返回结果；
- `print_summary(const std::vector<ImageResult>&)` 用只读引用传参，一个函数打印整张汇总表；
- 状态判断链：`!read_ok` → SKIP；`!saved` → FAIL；否则 OK。

## 构建与运行

```powershell
cd D:\opencv-learning\47_cpp_batch_image_processing
$env:MSYS2_ROOT = 'D:\VScode_C++\msys64'
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"
cmake --preset msys2-ucrt64
cmake --build --preset msys2-ucrt64
.\build\day47_batch_processing.exe .\assets .\output
```

## 最终验收记录

成功路径（`assets` 含 1 张正常图 + 1 张损坏图）：

```text
Found 2 image(s)
Result summary:
  SKIP (read-failed): .\assets\broken.jpg
  OK: .\output\day45_input_gray.jpg  427 x 640  type=0
Saved 1 / 2 image(s); read-failed 1; write-failed 0.
DAY47_BATCH_OK
```

写失败路径（`out2` 内含同名目录 `day45_input_gray.jpg`）：

```text
Result summary:
  SKIP (read-failed): .\assets\broken.jpg
  FAIL (write-failed): .\out2\day45_input_gray.jpg
Saved 0 / 2 image(s); read-failed 1; write-failed 1.
DAY47_BATCH_OK
```

退出码通过 `$LASTEXITCODE` 确认：

```text
正常路径退出码        = 0
无参数退出码          = 2
输入目录不存在退出码  = 1
输出目录被文件占用    = 3
```

## 关键记忆点

1. `create_directories` 幂等且能建多级目录；用不抛异常重载 + `std::error_code` 处理失败。
2. 用 OpenCV 模块前先确认 CMake 已链接：`cvtColor` → `imgproc`。
3. `imwrite` 返回 `bool`，写失败必须检查；OpenCV 的 WARN 日志不能替代代码判断。
4. `fs::path`：`operator/` 拼接路径，`stem()` 取主文件名；同主名不同扩展名会撞输出文件名。
5. 失败隔离分读/写两个维度，任何一步失败都只记录并继续，只有输出目录创建失败是致命错误。
6. `struct` 记录每张图的完整结果（含输出路径），`print_summary` 一个函数打印汇总表。
7. 灰度图 `type()==0`（`CV_8UC1`），彩色 `CV_8UC3==64`（OpenCV 5 编码）。

## Day48 计划（路线 A：加量）

- 主题：智能指针（`std::unique_ptr` / `std::shared_ptr`）+ 可配置处理管线；
- 产出：把 Day47 的批量处理器升级为 `--op` 可选算子（灰度 / 模糊 / 边缘）的可配置版本；
- 目标：9 月开学前每天 2~3 个知识点 + 一个可验收小程序，加快 C++ 阶段进度。
