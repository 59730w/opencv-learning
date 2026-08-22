# Day50：C++ 类封装、多文件工程与 CMake 多目标构建

日期：2026-08-22
学习阶段：C++ OpenCV 工程化过渡
前置内容：Day44～Day49，重点承接 Day48 的可配置图像处理管线与 Day49 的 Debug 构建经验

---

## 1. 今天解决什么问题

Day48 已经能使用一条命令批量执行 `gray`、`blur` 或 `edge`，但所有内容都放在一个 `.cpp` 文件中：

- 命令行参数解析；
- 文件系统遍历；
- OpenCV 图像处理；
- 输出文件命名；
- 结果汇总；
- `main()` 入口。

这种写法适合验证功能，却不适合继续增加线程、计时、测试和部署功能。文件越来越长以后，入口逻辑、业务逻辑和底层处理细节会互相干扰。

Day50 的目标不是学习新算法，而是保持 Day48 的外部行为不变，把程序重构为：

```text
命令行程序 day50_oop_pipeline
              │
              │ 使用公开接口
              ▼
静态库 day50_pipeline
              │
              │ 调用
              ▼
OpenCV core / imgcodecs / imgproc
```

完成后，`main()` 只负责“准备参数并调用管线”，图像处理细节由 `ImagePipeline` 类负责。

---

## 2. 今日学习成果

本节完成了以下内容：

1. 使用 `class ImagePipeline` 封装处理操作和处理行为；
2. 区分 `public` 接口与 `private` 实现；
3. 使用构造函数和成员初始化列表保存管线配置；
4. 使用 `const` 成员函数表达“不修改对象状态”；
5. 使用 `[[nodiscard]]` 提醒调用者不要忽略重要返回值；
6. 将类声明放入 `.hpp`，将实现放入 `.cpp`；
7. 使用普通命名空间和匿名命名空间控制名称可见范围；
8. 使用 CMake 创建静态库和可执行程序两个目标；
9. 理解 `PUBLIC` 与 `PRIVATE` 依赖的基本传播规则；
10. 保留 Day48 的三种操作和退出码契约；
11. 采用测试先行方式完成 35 项自动验收；
12. 对灰度、模糊和边缘检测结果进行了实际视觉检查。

---

## 3. 目录结构与文件职责

```text
50_cpp_oop_multifile_cmake/
├── .gitignore
├── CMakeLists.txt
├── CMakePresets.json
├── assets/
│   └── input/
│       └── day45_input.jpg
├── include/
│   └── day50/
│       └── image_pipeline.hpp
├── src/
│   └── image_pipeline.cpp
├── code/
│   ├── day50_oop_multifile.cpp
│   └── day50_notes.md
└── tests/
    └── day50_contract.ps1
```

### 3.1 `include/day50/image_pipeline.hpp`

这是公开头文件，负责声明：

- `Operation`：可选处理类型；
- `ImageResult`：单张图片的处理结果；
- 操作名解析和结果汇总函数；
- `ImagePipeline` 类的公开接口和私有成员。

调用者阅读这个文件就能知道库“可以做什么”，不需要先理解全部实现。

### 3.2 `src/image_pipeline.cpp`

这是实现文件，负责：

- 判断图片扩展名；
- 遍历和排序输入图片；
- 灰度、模糊和边缘处理；
- 生成输出文件名；
- 单图读取、处理和写入；
- 汇总处理结果。

### 3.3 `code/day50_oop_multifile.cpp`

这是程序入口，只负责：

1. 检查命令行格式；
2. 解析处理类型；
3. 检查输入和输出目录；
4. 创建 `ImagePipeline` 对象；
5. 调用公开接口；
6. 打印结果和成功标记。

这里没有 `cv::cvtColor`、`cv::GaussianBlur`、`cv::Canny` 或 `cv::imwrite`，说明入口程序不再依赖图像处理细节。

### 3.4 `tests/day50_contract.ps1`

这是自动契约测试，验证：

- 文件和构建目标存在；
- 类确实具有公开与私有区域；
- 主程序没有重新包含 OpenCV 处理细节；
- CMake 建立了库与可执行程序的依赖关系；
- 三种模式都能运行和落盘；
- 三张结果图内容不同；
- 退出码和成功标记稳定。

---

## 4. 类、对象与封装

### 4.1 类与对象的区别

类是类型设计，对象是按照该类型创建出来的实例。

```cpp
class ImagePipeline {
    // 类型的定义
};

ImagePipeline pipeline{Operation::Gray};
```

这里：

- `ImagePipeline` 是类；
- `pipeline` 是对象；
- `Operation::Gray` 是这个对象保存的配置。

可以把类理解为图纸，把对象理解为根据图纸制造出的具体设备。

### 4.2 封装不只是“把函数放进类里”

真正的封装包含两个方面：

1. 把相关状态和行为放在一起；
2. 隐藏调用者不应该直接控制的细节。

本节中，处理类型 `operation_` 与 `apply_operation()`、`process_image()` 等行为属于同一条图像处理管线，所以适合放在一个类中。

如果只是把所有自由函数机械地改成成员函数，但所有成员仍然公开，调用者仍然能随意破坏内部流程，那么并没有形成有效封装。

### 4.3 `public` 与 `private`

```cpp
class ImagePipeline {
public:
    explicit ImagePipeline(Operation operation) noexcept;
    Operation operation() const noexcept;
    std::vector<ImageResult> process_directory(...) const;

private:
    cv::Mat apply_operation(...) const;
    ImageResult process_image(...) const;
    Operation operation_;
};
```

`public` 是调用者可以使用的稳定接口：

- 如何创建管线；
- 如何查看当前操作；
- 如何处理一个目录。

`private` 是类自己管理的实现细节：

- 单张图如何处理；
- 输出名称如何生成；
- 当前操作存在哪里。

这样做的好处是，以后即使更换内部实现，只要公开接口不变，`main()` 通常不需要修改。

---

## 5. 构造函数与对象有效性

构造函数在对象创建时自动执行：

```cpp
ImagePipeline::ImagePipeline(Operation operation) noexcept
    : operation_(operation) {
}
```

`operation_(operation)` 是成员初始化列表，它直接初始化成员变量，而不是先默认初始化再赋值。

创建对象：

```cpp
ImagePipeline pipeline{*operation};
```

这意味着：

- 不提供操作类型就无法创建管线；
- 对象创建完成后立即拥有处理配置；
- 不需要再调用 `set_operation()` 补充状态；
- 后续成员函数不用反复接收同一个 `operation` 参数。

### 5.1 为什么使用 `explicit`

构造函数只有一个参数时，C++ 可能允许从这个参数隐式转换成对象。`explicit` 表示必须明确写出对象构造，减少意外转换：

```cpp
ImagePipeline pipeline{Operation::Gray};  // 清晰
```

### 5.2 为什么使用 `noexcept`

这个构造函数只保存一个枚举值，不需要分配资源，也没有预期异常，因此标记 `noexcept`，向调用者和编译器说明它不会抛出异常。

注意：不能为了“看起来安全”而给所有函数加 `noexcept`。如果函数内部可能抛异常而又标记了 `noexcept`，异常逃出时程序会终止。

---

## 6. `const` 成员函数

成员函数末尾的 `const` 表示该函数不会修改对象的可观察状态：

```cpp
Operation operation() const noexcept;

std::vector<ImageResult> process_directory(
    const fs::path& input_dir,
    const fs::path& output_dir) const;
```

虽然 `process_directory()` 会读取图片并写出新文件，但它不会修改 `pipeline.operation_`，因此仍然可以是 `const` 成员函数。

这里需要区分：

- “不修改对象状态”；
- “完全没有外部副作用”。

`const` 只保证前者，不保证函数不会读写文件。

---

## 7. `[[nodiscard]]` 的作用

以下返回值很重要：

```cpp
[[nodiscard]] Operation operation() const noexcept;
[[nodiscard]] std::vector<ImageResult> process_directory(...) const;
```

如果调用者完全忽略处理结果，可能错过读取或写入失败。`[[nodiscard]]` 会让编译器对不合理的忽略行为发出警告。

它不是运行时检查，也不能强制调用者正确处理每一种失败；它是一种编译期接口提示。

---

## 8. 头文件与源文件分离

### 8.1 头文件放声明

```cpp
class ImagePipeline {
public:
    explicit ImagePipeline(Operation operation) noexcept;
};
```

声明回答：“这个类具有什么能力？”

### 8.2 源文件放定义

```cpp
ImagePipeline::ImagePipeline(Operation operation) noexcept
    : operation_(operation) {
}
```

定义回答：“这个能力具体如何实现？”

### 8.3 `#pragma once`

头文件第一行使用：

```cpp
#pragma once
```

它防止同一翻译单元中重复包含头文件，避免类和结构体重复定义。

### 8.4 为什么实现文件先包含自己的头文件

```cpp
#include "day50/image_pipeline.hpp"
```

把自己的头文件放在最前面，有助于发现头文件是否遗漏了它真正依赖的标准库或 OpenCV 声明。如果只有借助其他头文件的间接包含才能编译，代码会非常脆弱。

---

## 9. 命名空间

### 9.1 普通命名空间

公开接口放在：

```cpp
namespace day50 {
    // public declarations
}
```

调用时写：

```cpp
day50::parse_operation(...);
day50::ImagePipeline pipeline{...};
```

这样能避免以后其他课程或第三方库也定义 `ImagePipeline`、`Operation` 时发生名称冲突。

### 9.2 匿名命名空间

只在 `image_pipeline.cpp` 内部使用的辅助函数放入：

```cpp
namespace {
    bool is_image_extension(...);
    std::vector<fs::path> list_images(...);
}
```

这些函数不会成为库的公开接口，也不会与其他源文件中的同名函数冲突。

---

## 10. `ImagePipeline` 的运行流程

```text
main()
  │
  ├─ parse_operation("gray")
  │
  ├─ ImagePipeline pipeline{Operation::Gray}
  │
  └─ pipeline.process_directory(input, output)
       │
       ├─ list_images(input)
       │
       ├─ process_image(image, output)
       │    ├─ cv::imread
       │    ├─ apply_operation
       │    ├─ output_path_for
       │    └─ cv::imwrite
       │
       └─ 返回 vector<ImageResult>
```

主程序最后调用 `print_summary()`，根据每个 `ImageResult` 输出：

- 读取失败；
- 写入失败；
- 成功及输出尺寸。

---

## 11. `cv::Mat` 与本节的对象设计

`apply_operation()` 使用：

```cpp
cv::Mat apply_operation(const cv::Mat& color) const;
```

输入使用 `const cv::Mat&`：

- 避免不必要的 `cv::Mat` 头部复制；
- 明确函数不修改输入图像；
- 允许处理较大的图像而不复制像素数据。

输出按值返回 `cv::Mat`。`cv::Mat` 本身使用引用计数管理像素缓冲区，按值返回通常不会复制整张图的像素数据，并且现代 C++ 还可以进行返回值优化或移动。

本节没有手动使用 `new`、`delete` 或裸指针。对象和局部变量都通过作用域自动管理生命周期。

---

## 12. CMake 多目标设计

### 12.1 两个目标

```cmake
add_library(day50_pipeline STATIC
    src/image_pipeline.cpp
)

add_executable(day50_oop_pipeline
    code/day50_oop_multifile.cpp
)
```

生成结果：

```text
build/libday50_pipeline.a
build/day50_oop_pipeline.exe
```

静态库是一组已经编译的目标文件归档，本身不是可以直接运行的程序。可执行程序链接静态库后，才能获得 `ImagePipeline` 的实现。

### 12.2 目标之间的链接

```cmake
target_link_libraries(day50_oop_pipeline
    PRIVATE
        day50_pipeline
)
```

这条语句同时表达两件事：

1. `day50_oop_pipeline` 构建时依赖 `day50_pipeline`；
2. 链接可执行程序时要加入该静态库。

因此 Ninja 会自动按正确顺序执行：

```text
编译 image_pipeline.cpp
        ↓
生成 libday50_pipeline.a
        ↓
编译 day50_oop_multifile.cpp
        ↓
链接 day50_oop_pipeline.exe
```

### 12.3 `target_include_directories`

```cmake
target_include_directories(day50_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
```

`PUBLIC` 表示：

- 编译 `day50_pipeline` 自己需要这些头文件目录；
- 链接 `day50_pipeline` 的目标也需要这些目录。

主程序因此可以直接写：

```cpp
#include "day50/image_pipeline.hpp"
```

而不需要再次手工配置 `include/` 路径。

### 12.4 OpenCV 依赖的 `PUBLIC` 与 `PRIVATE`

```cmake
target_link_libraries(day50_pipeline
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgcodecs
        opencv_imgproc
)
```

本节的理解方式：

- 公开头文件中出现了 `cv::Mat`，调用者编译头文件时需要 OpenCV Core，因此 `opencv_core` 属于公开使用需求；
- `cv::imread`、`cv::imwrite`、`cv::cvtColor`、`GaussianBlur` 和 `Canny` 只出现在实现文件中，因此 imgcodecs 和 imgproc 是实现依赖。

`PUBLIC` 和 `PRIVATE` 描述的是“依赖如何传播”，不是“谁更重要”。

---

## 13. 构建环境与命令

本节验证环境：

```text
编译器：MSYS2 UCRT64 G++ 16.2.0
CMake：4.4.2
生成器：Ninja 1.13.2
OpenCV：5.0.0
C++ 标准：C++17
构建类型：Debug
```

PowerShell 中配置环境：

```powershell
$env:MSYS2_ROOT = "D:\VScode_C++\msys64"
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"
```

进入课程目录：

```powershell
Set-Location D:\opencv-learning\50_cpp_oop_multifile_cmake
```

配置项目：

```powershell
cmake --preset msys2-ucrt64-debug
```

干净构建：

```powershell
cmake --build --preset msys2-ucrt64-debug --clean-first
```

构建时启用了：

```text
-Wall -Wextra -Wpedantic
```

这些选项用于尽早暴露可疑代码。警告不是错误，但学习阶段不应在不了解原因的情况下忽略警告。

---

## 14. 运行命令

### 14.1 灰度处理

```powershell
.\build\day50_oop_pipeline.exe `
  .\assets\input `
  .\outputs\gray `
  --op gray
```

### 14.2 高斯模糊

```powershell
.\build\day50_oop_pipeline.exe `
  .\assets\input `
  .\outputs\blur `
  --op blur
```

### 14.3 Canny 边缘检测

```powershell
.\build\day50_oop_pipeline.exe `
  .\assets\input `
  .\outputs\edge `
  --op edge
```

成功运行时会输出：

```text
Operation: gray
Found 1 image(s)
Result summary:
  OK: ...  427 x 640  type=0
Saved 1 / 1 image(s); read-failed 0; write-failed 0.
DAY50_OOP_OK
```

不同操作的 `type` 可能不同：

- 灰度和边缘结果通常是单通道 `CV_8UC1`；
- 模糊结果保留 BGR 三通道，通常是 `CV_8UC3`。

不要只依赖 `type` 数字记忆图像含义，应同时结合通道数、数据深度和创建结果的处理流程理解。

---

## 15. 退出码契约

| 退出码 | 含义 | 示例 |
| ---: | --- | --- |
| `0` | 程序流程完成 | 输入和输出目录有效 |
| `1` | 输入目录无效 | 路径不存在或不是目录 |
| `2` | 命令行格式错误 | 参数缺失或 `--op` 位置错误 |
| `3` | 输出目录无效 | 输出路径是普通文件或无法创建 |
| `4` | 未知操作 | 使用 `--op rotate` |

退出码使脚本和上层程序可以在不解析自然语言日志的情况下判断失败类型。

需要注意：当前版本遇到单张图片读取失败或写入失败时，会在 `ImageResult` 中记录并继续处理其他图片。程序整体仍然返回 `0`。这是批处理的“单文件失败隔离”策略，不代表所有文件都成功，调用者仍应查看汇总。

---

## 16. 测试驱动过程

本节遵循 RED → GREEN → REFACTOR：

### 16.1 RED

先创建 `tests/day50_contract.ps1`，当头文件和实现尚不存在时执行。

得到预期失败：

```text
[FAIL] public header exists
```

这证明测试能真实发现缺失功能，而不是在实现完成后立即通过的“装饰性测试”。

### 16.2 GREEN

实现最小的头文件、类、主程序和 CMake 目标，随后干净构建并运行同一份脚本。

首次 GREEN 结果：

```text
Passed 35 Day50 checks.
DAY50_CONTRACT_OK
```

### 16.3 REFACTOR

在测试全绿后：

- 补充 `<cctype>` 显式依赖；
- 删除未使用的 `<utility>`；
- 保持公开接口和行为不变；
- 再次执行完整构建和测试。

---

## 17. 自动验收覆盖范围

运行：

```powershell
.\tests\day50_contract.ps1
```

测试共覆盖 35 项，包括：

1. 公开头文件存在；
2. 类实现文件存在；
3. 精简主程序存在；
4. CMake 文件存在；
5. Debug 可执行程序存在；
6. 静态库存在；
7. 声明了 `ImagePipeline`；
8. 存在 `public` 接口；
9. 存在 `private` 实现；
10. 构造函数在源文件中定义；
11. 主程序创建管线对象；
12. 主程序没有图像处理细节；
13. CMake 创建静态库；
14. CMake 创建可执行程序；
15. 可执行程序链接静态库；
16～20. 灰度模式的退出码、日志、保存、标记和文件；
21～25. 模糊模式的退出码、日志、保存、标记和文件；
26～30. 边缘模式的退出码、日志、保存、标记和文件；
31. 三种输出哈希互不相同；
32. 输入目录不存在时返回 `1`；
33. 参数缺失时返回 `2`；
34. 输出路径不是目录时返回 `3`；
35. 操作名未知时返回 `4`。

哈希不同只能证明文件内容不同，不能证明视觉结果一定合理，所以仍然需要实际查看图片。

---

## 18. 输出图像视觉检查

### 18.1 Gray

- 输出为灰度图；
- 树干、树冠、地面和天空层次仍然可辨认；
- 没有出现全黑、全白或尺寸异常。

### 18.2 Blur

- 保留彩色信息；
- 树枝、草地纹理和远处轮廓明显变柔和；
- 符合 `7 x 7` 高斯核、`sigma=1.5` 的预期效果。

### 18.3 Edge

- 背景为黑色，主要边缘为白色；
- 树干、树枝、地平线和部分草地纹理被检测出来；
- 天空等平坦区域边缘较少；
- 符合阈值 `80 / 160` 的 Canny 结果特征。

三张图片尺寸一致，但通道数和像素含义不同。

---

## 19. 常见错误与排查顺序

### 19.1 找不到 `day50/image_pipeline.hpp`

典型原因：

- 没有配置 `${CMAKE_CURRENT_SOURCE_DIR}/include`；
- `target_include_directories()` 作用在了错误目标；
- 文件路径或大小写错误。

先检查：

```cmake
target_include_directories(day50_pipeline PUBLIC ...)
```

### 19.2 `undefined reference to ImagePipeline...`

编译阶段能找到声明，但链接阶段找不到实现。

检查：

- `src/image_pipeline.cpp` 是否加入 `add_library()`；
- 函数签名是否与头文件完全一致；
- 实现是否位于 `namespace day50`；
- executable 是否链接 `day50_pipeline`。

### 19.3 `undefined reference` 指向 OpenCV

检查静态库目标是否链接：

```text
opencv_core
opencv_imgcodecs
opencv_imgproc
```

### 19.4 修改头文件后没有重新编译

正常情况下 Ninja 会追踪头文件依赖并重编译相关目标。如果行为异常，可执行：

```powershell
cmake --build --preset msys2-ucrt64-debug --clean-first
```

### 19.5 输出目录参数指向了图片文件

如果输出路径已经存在但不是目录，程序返回退出码 `3`。这避免把文件路径误当成目录继续处理。

### 19.6 CMake 修改后构建结果不符合预期

先重新配置，再构建：

```powershell
cmake --preset msys2-ucrt64-debug
cmake --build --preset msys2-ucrt64-debug --clean-first
```

---

## 20. 六个核心问题

### 问题一：类解决了什么问题？

类把处理配置 `operation_` 与依赖该配置的处理行为组织在一起，并通过公开接口限制调用方式。它降低了 `main()` 与 OpenCV 处理细节的耦合，使后续增加计时或并行实现时不必重写命令行入口。

### 问题二：封装为什么不等于把函数塞进类里？

因为封装的关键是建立边界和不变量。调用者只看到必要接口，内部状态不能被随意修改，内部步骤可以独立变化。如果所有成员都公开，或者类没有需要维护的状态和边界，机械迁移函数并不会自动提高设计质量。

### 问题三：构造函数如何保证对象有效？

`ImagePipeline` 必须在创建时接收一个 `Operation`，并通过初始化列表保存到 `operation_`。对象不存在“先创建但还没配置”的中间状态，因此创建后即可安全调用 `process_directory()`。

### 问题四：为什么主程序不需要知道图像处理细节？

主程序只依赖 `ImagePipeline` 的公开接口。灰度、模糊和边缘的具体 OpenCV 调用都在实现文件中。只要接口和行为契约不变，内部算法可以修改，而不影响入口程序。

### 问题五：为什么分别创建 library 和 executable target？

库目标承载可复用的图像处理能力，可执行目标承载一种具体入口。分开后，未来可以让测试程序、GUI、另一个命令行工具或嵌入式入口复用同一个库，而不用复制源代码。

### 问题六：`PUBLIC` 与 `PRIVATE` 如何影响依赖传播？

`PUBLIC` 需求既用于构建当前目标，也传播给链接它的目标；`PRIVATE` 需求只描述当前目标自己的实现需要。选择依据是依赖是否出现在公开接口或是否是调用者编译接口所必需，而不是简单地把所有依赖都写成 `PUBLIC`。

---

## 21. 本节设计的优点与限制

### 优点

- 入口与处理逻辑分离；
- 公开接口较小；
- 类对象创建后立即有效；
- OpenCV 处理细节集中；
- 静态库可以被其他目标复用；
- 自动测试覆盖正常路径和主要错误路径；
- 保留批处理中的单文件失败隔离。

### 当前限制

- 仍然是单线程顺序处理；
- 模糊核和 Canny 阈值是固定值；
- 扩展名判断不能识别无扩展名图片；
- 没有递归扫描子目录；
- 没有单独的 C++ 单元测试框架；
- 没有安装、导出或打包 CMake 目标；
- 整体退出码 `0` 不等于每一张图片都处理成功，需要结合汇总判断。

这些限制是有意保留的。Day50 只解决 OOP、多文件和目标化构建，不提前混入 Day51 的并行内容或 Day52 的测试与打包内容。

---

## 22. 与机器视觉项目的联系

真实机器视觉程序通常不会把相机采集、图像预处理、模型推理、结果展示和硬件通信全部写进 `main()`。

Day50 建立的基本分层可以继续扩展为：

```text
应用入口
├── 图像采集模块
├── 图像预处理模块
├── 模型推理模块
├── 结果后处理模块
└── 输出或硬件控制模块
```

每个模块拥有小而清晰的接口，CMake 用目标描述模块之间的依赖。这正是后续嵌入式视觉、ONNX Runtime 推理和导师项目中会反复使用的工程思路。

---

## 23. 今日核心记忆点

1. 类是状态与行为的边界，不是函数收纳盒。
2. `public` 表示稳定使用方式，`private` 表示内部实现细节。
3. 构造函数应让对象创建后立即处于有效状态。
4. 成员初始化列表比“先默认构造再赋值”更直接。
5. 成员函数末尾的 `const` 表示不修改对象状态，不代表没有文件 I/O。
6. 头文件声明能力，源文件实现能力。
7. 匿名命名空间适合隐藏只在一个 `.cpp` 中使用的辅助函数。
8. CMake target 是编译选项、头文件路径和链接依赖的承载单位。
9. 静态库不能直接运行，要链接进可执行程序。
10. `PUBLIC` 和 `PRIVATE` 描述依赖传播边界。
11. 自动测试证明行为稳定，视觉检查证明图片结果合理，两者不能互相替代。
12. `build/`、`outputs/`、可执行文件和静态库都是生成物，不应上传 GitHub。

---

## 24. Day51 预告

下一步计划学习：

- `std::thread` 的基本线程模型；
- OpenCV `cv::parallel_for_`；
- `std::chrono` 运行时间测量；
- 单线程与并行批处理结果一致性；
- 小规模性能对比和正确解释加速结果。

Day51 会继续复用 Day50 的 `ImagePipeline` 分层，让并行实现集中在库内部，而不是重新把逻辑堆回 `main()`。
