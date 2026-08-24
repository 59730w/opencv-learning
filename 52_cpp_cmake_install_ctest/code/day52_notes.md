# Day52：CMake 安装、导出、CTest 单元测试与 CPack 打包

## 1. 今日定位

Day44-Day51 已经依次完成了 C++/OpenCV 环境、`cv::Mat` 所有权、批处理、异常隔离、智能指针、GDB、类封装、静态库、多线程和性能基准。Day52 不再增加新的图像算法，而是回答一个工程问题：

> 已经写好的 C++ 图像处理库，怎样测试、安装、提供给另一个工程使用，并打成一个结构正确的开发包？

今天继续复用灰度、模糊和边缘检测，因为它们已经足够稳定，可以充当 CMake 和测试技术的载体。学习重点是：

1. 让容易出错的策略逻辑可以被真正的 C++ 单元测试直接调用；
2. 用 CTest 统一发现和运行测试；
3. 区分源码树、构建树和安装树；
4. 用 CMake 目标表达“这个库应该怎样被使用”；
5. 导出 `day52::pipeline`，让另一个工程通过 `find_package()` 使用它；
6. 用 CPack 把安装内容生成 ZIP；
7. 验证 ZIP 的内容和可迁移性边界。

今天没有引入 Catch2 或 GoogleTest。原因不是它们不好，而是今天首先要理解“测试程序 + 非零退出码 + CTest 调度”的最小闭环。等进入更大项目后，再引入成熟测试框架会更容易理解其价值。

---

## 2. 今日完成结果

实际环境：

- Windows 11；
- MSYS2 UCRT64；
- G++ 16.2.0；
- CMake 4.4.2；
- Ninja 1.13.2；
- OpenCV 5.0.0；
- C++17；
- Debug 构建。

最终验证结果：

- `day52_pipeline` 静态库构建成功；
- `day52_parallel_pipeline` 主程序构建成功；
- `day52_unit_tests` 单元测试程序构建成功；
- 21 项 C++ 单元断言全部通过；
- 60 项 PowerShell 端到端契约检查全部通过；
- CTest 中 2/2 个测试通过；
- 安装树生成成功，共 7 个安装文件；
- 安装树消费者构建并运行成功；
- CPack ZIP 生成成功；
- 13 项安装包结构检查全部通过；
- 从解压后的 ZIP 再次配置消费者成功；
- 解压包内主程序处理 4/4 张图片成功。

确定性成功标记：

```text
DAY52_UNIT_TESTS_OK
DAY52_CONTRACT_OK
DAY52_CONSUMER_OK
DAY52_PACKAGE_OK
DAY52_PIPELINE_OK
```

按脚本内部断言统计，今天至少执行了：

```text
21 + 60 + 13 = 94 项自动检查
```

另外还执行了安装树消费者、ZIP消费者和包内可执行程序的实际运行验证。

---

## 3. 项目目录结构

源码目录：

```text
52_cpp_cmake_install_ctest/
├── .gitignore
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
│   └── day52Config.cmake.in
├── assets/
│   └── input/
│       ├── banana_256x256.png
│       ├── dog_400x301.jpg
│       ├── hotdog_74x119.png
│       └── personal_563x845.jpg
├── include/
│   └── day52/
│       └── image_pipeline.hpp
├── src/
│   └── image_pipeline.cpp
├── code/
│   ├── day52_cmake_install_ctest.cpp
│   └── day52_notes.md
└── tests/
    ├── day52_unit_tests.cpp
    ├── day52_contract.ps1
    ├── day52_package_check.ps1
    └── consumer/
        ├── CMakeLists.txt
        └── main.cpp
```

各目录职责：

- `include/day52/`：公开头文件，安装后给消费者编译使用；
- `src/`：库的实现，不属于公开接口；
- `code/`：课程主程序和笔记；
- `tests/`：单元测试、契约测试、包检查和独立消费者；
- `cmake/`：生成安装包配置所需的模板；
- `assets/input/`：少量、可审查、可提交的真实测试图片；
- `build/`：所有生成物，已被 `.gitignore` 忽略。

---

## 4. 实验图片与来源

今天没有复制整个数据集，只选择了 4 张格式和尺寸不同的图片：

| 仓库内文件 | 原始路径 | 格式 | 尺寸 | 用途 |
| --- | --- | --- | --- | --- |
| `personal_563x845.jpg` | `D:\images\test.jpg` | JPEG RGB | 563×845 | 较大的普通照片 |
| `banana_256x256.png` | `D:\DL_code\data\banana-detection\bananas_train\images\0.png` | PNG RGB | 256×256 | 正方形检测数据图片 |
| `hotdog_74x119.png` | `D:\DL_code\data\hotdog\test\hotdog\1000.png` | PNG RGB | 74×119 | 很小的输入图片 |
| `dog_400x301.jpg` | `D:\DL_code\data\kaggle_dog_tiny\test\0dc570ec7086bab004a7e357164c04b8.jpg` | JPEG RGB | 400×301 | 非正方形分类数据图片 |

多样输入可以检查：

- JPEG 和 PNG 是否都能读取；
- 大图和小图是否都能处理；
- 输出尺寸是否保持一致；
- 多线程任务数是否会被实际图片数量限制。

这些图片只是程序功能测试素材，不用于训练，也不能用于任何模型泛化结论。

---

## 5. 源码树、构建树和安装树

这是今天最核心的概念。

### 5.1 源码树 source tree

源码树是人工编写和需要提交到 Git 的内容：

```text
CMakeLists.txt
include/
src/
code/
tests/
assets/
```

源码树回答：“项目由哪些源文件组成？”

### 5.2 构建树 build tree

构建树由 CMake 和编译器生成：

```text
build/
├── CMakeCache.txt
├── build.ninja
├── CTestTestfile.cmake
├── day52_parallel_pipeline.exe
├── day52_unit_tests.exe
├── libday52_pipeline.a
└── ...
```

它的目录结构主要服务于当前构建工具，不保证适合发布，也不应该提交 Git。

### 5.3 安装树 install tree

安装树由 `cmake --install` 根据 `install()` 规则生成：

```text
build/install/
├── bin/
│   └── day52_parallel_pipeline.exe
├── include/
│   └── day52/
│       └── image_pipeline.hpp
└── lib/
    ├── libday52_pipeline.a
    └── cmake/
        └── day52/
            ├── day52Config.cmake
            ├── day52ConfigVersion.cmake
            ├── day52Targets.cmake
            └── day52Targets-debug.cmake
```

安装树回答：“另一个程序真正需要拿到哪些文件，以及应该到哪里寻找它们？”

### 5.4 `build` 和 `install` 的区别

```text
cmake --build build
```

表示编译当前项目，输出位置由生成器决定。

```text
cmake --install build
```

表示执行 `CMakeLists.txt` 中写好的安装规则，把选定的目标、头文件和配置重新组织成对外结构。

安装不是再次编译，也不是简单复制整个 `build/`。

---

## 6. 真正的 C++ 单元测试

Day51 的 PowerShell 脚本主要从命令行运行整个程序。它能证明程序对用户来说能工作，但当某个内部规则出错时，定位范围仍然较大。

Day52 将工作线程选择逻辑提取成公开纯函数：

```cpp
unsigned int resolve_worker_count(
    ExecutionMode mode,
    unsigned int requested_workers,
    std::size_t image_count);
```

所谓纯函数，在这里指：

- 输入完全来自参数；
- 不读写图片；
- 不遍历目录；
- 不真正创建线程；
- 相同输入得到相同输出。

因此单元测试可以非常直接：

```cpp
expect(
    day52::resolve_worker_count(
        ExecutionMode::Threads, 8, 3) == 3,
    "worker count is capped by image count");
```

测试覆盖了：

1. `gray/blur/edge` 的解析；
2. 非法操作和空字符串；
3. `sequential/threads` 的解析；
4. 非法执行模式；
5. 枚举值反向转成字符串；
6. 顺序模式固定为 1 个工作线程；
7. 线程数不超过图片数；
8. 空目录和单张图片的安全行为；
9. 请求 0 个线程时抛出 `std::invalid_argument`；
10. `ImagePipeline` 构造函数拒绝 0 个线程。

测试程序维护成功和失败计数。只要任意断言失败，`main()` 返回 1；全部通过则返回 0，并输出：

```text
Passed 21 Day52 unit checks.
DAY52_UNIT_TESTS_OK
```

退出码非常重要，因为 CTest 不需要理解所有日志，只需要知道进程是否成功。

---

## 7. 单元测试、契约测试、消费者测试和包测试

今天有四个不同测试层次。

| 测试类型 | 直接测试什么 | 能发现什么 | 不能替代什么 |
| --- | --- | --- | --- |
| C++ 单元测试 | 单个函数、异常和策略 | 解析错误、边界条件、策略错误 | 不能证明命令行和文件处理可用 |
| PowerShell 契约测试 | 完整 EXE、参数、文件和退出码 | 参数协议、输入输出、损坏文件隔离、顺序/线程一致性 | 不能精确定位所有内部函数错误 |
| 消费者测试 | 安装后的 `day52::pipeline` | 导出目标、头文件、依赖传播、`find_package()` | 不能证明 ZIP 内容完整 |
| 包测试 | 安装树和 ZIP | 缺文件、硬编码源码路径、导出目标名错误 | 不能替代算法和命令行测试 |

工程测试通常不是“选一个最强测试”，而是让不同层次共同形成证据链：

```text
函数正确
  ↓
完整程序正确
  ↓
安装后的库可以被复用
  ↓
打包内容完整
```

---

## 8. CTest 是什么

CTest 是 CMake 提供的测试调度工具。它不强制使用某个 C++ 测试框架；只要有一个可以运行并返回退出码的命令，就可以注册为测试。

顶层 CMake 使用：

```cmake
include(CTest)
```

它会提供 `BUILD_TESTING` 选项，默认值为 `ON`。然后仅在启用测试时构建测试目标：

```cmake
if(BUILD_TESTING)
    add_executable(day52_unit_tests
        tests/day52_unit_tests.cpp
    )

    target_link_libraries(day52_unit_tests
        PRIVATE day52_pipeline
    )

    add_test(
        NAME day52_unit_tests
        COMMAND day52_unit_tests
    )
endif()
```

这里要区分：

- `add_executable()`：告诉 CMake 怎样构建测试程序；
- `add_test()`：告诉 CTest 怎样运行测试程序。

只有第一句，没有第二句，测试程序虽然被编译出来，但 `ctest` 不会发现它。

今天还把 PowerShell 契约测试注册进 CTest，因此一条命令可以运行两种测试：

```powershell
ctest --preset msys2-ucrt64-debug -V
```

最终结果：

```text
1/2 Test #1: day52_unit_tests ... Passed
2/2 Test #2: day52_contract ..... Passed

100% tests passed out of 2
```

虽然 CTest 显示 2 个测试，但两个测试程序内部一共执行了 81 项断言检查。

---

## 9. PowerShell 端到端契约测试

`tests/day52_contract.ps1` 从 CTest 接收主程序的真实路径：

```cmake
-Executable $<TARGET_FILE:day52_parallel_pipeline>
```

`$<TARGET_FILE:...>` 是生成器表达式。它让 CMake 根据实际平台和构建方式找到可执行文件，而不是在脚本中猜测路径。

契约测试验证了：

- 源码结构和关键 CMake 声明存在；
- 4 张代表性图片存在；
- `gray`、`blur`、`edge` 均能处理 4/4 张图片；
- 顺序模式有效线程数为 1；
- 请求 8 个线程、只有 4 张图时，有效线程数被限制为 4；
- 顺序边缘结果和多线程边缘结果的 SHA256 完全一致；
- 输出 JPEG 可以再次被 OpenCV 读取；
- 1 张损坏图片不会阻止其余 4 张图片输出；
- 空目录不会发生除零；
- 退出码 1-6 的错误分支都符合约定。

退出码约定：

| 退出码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 1 | 输入目录不存在或不是目录 |
| 2 | 命令行参数结构错误 |
| 3 | 输出目录无效或无法创建 |
| 4 | 未知图像操作 |
| 5 | 未知执行模式 |
| 6 | 工作线程数非法 |
| 7 | 处理阶段捕获到异常 |

最终输出：

```text
Passed 60 Day52 contract checks.
DAY52_CONTRACT_OK
```

---

## 10. PUBLIC、PRIVATE 和 INTERFACE

现代 CMake 尽量围绕“目标”组织信息，而不是到处写全局头文件目录和库路径。

```cmake
target_link_libraries(day52_pipeline
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgcodecs
        opencv_imgproc
        Threads::Threads
)
```

三种可见性可以这样理解：

### `PRIVATE`

当前目标自己需要，消费者不直接使用。

例如 `image_pipeline.cpp` 内部调用 `cv::imread`、`cv::Canny` 和线程实现，所以这些是实现依赖。

### `PUBLIC`

当前目标自己需要，消费者也需要知道。

公开头文件中出现了 `cv::Mat`，消费者编译这个头文件时必须能找到 OpenCV Core 的头文件，所以 `opencv_core` 是公开使用要求。

### `INTERFACE`

当前目标自己不使用，但使用它的消费者需要。它常用于纯头文件库或只负责传播编译选项的接口目标。

对于静态库，即使某些链接依赖写成 `PRIVATE`，最终消费者链接时仍可能需要对应库。CMake 导出的目标会保存必要的链接信息，包配置则先用 `find_dependency()` 创建这些外部依赖目标。

---

## 11. BUILD_INTERFACE 与 INSTALL_INTERFACE

Day51 使用的是源码目录头文件：

```text
D:/opencv-learning/51_cpp_threads_timing/include
```

如果把这个绝对路径直接写入导出的目标，那么把安装包复制到其他目录后，它仍会回头寻找原源码目录，包就不是真正可迁移的。

Day52 使用生成器表达式：

```cmake
target_include_directories(day52_pipeline
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
```

含义：

- 在当前源码/构建树使用库时，头文件位于源码的 `include/`；
- 安装后使用库时，头文件位于安装前缀下的 `include/`；
- 同一个 CMake 目标可以在两个阶段拥有正确的使用方式。

这不是普通环境变量替换，而是 CMake 在生成和导出目标时根据上下文解释的表达式。

---

## 12. GNUInstallDirs

顶层 CMake 使用：

```cmake
include(GNUInstallDirs)
```

它提供标准安装目录变量：

| 变量 | 当前常见值 | 内容 |
| --- | --- | --- |
| `CMAKE_INSTALL_BINDIR` | `bin` | 可执行程序和运行时文件 |
| `CMAKE_INSTALL_LIBDIR` | `lib` | 静态库、导入库和包配置 |
| `CMAKE_INSTALL_INCLUDEDIR` | `include` | 公开头文件 |

使用标准变量比手写目录更清晰，也更容易适配其他平台和工具链。

---

## 13. install(TARGETS)

核心安装规则：

```cmake
install(
    TARGETS
        day52_pipeline
        day52_parallel_pipeline
    EXPORT day52Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

主要类别：

- `RUNTIME`：Windows 的 `.exe`、运行时 DLL 等；
- `ARCHIVE`：静态库 `.a/.lib` 和 Windows 导入库；
- `LIBRARY`：Linux 共享库等非运行时动态库；
- `INCLUDES`：声明导出目标的安装头文件使用目录。

公开头文件单独安装：

```cmake
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

测试程序没有出现在 `install(TARGETS)` 中，因此不会进入开发包。这是有意的：测试属于项目开发过程，用户使用库通常不需要测试 EXE。

---

## 14. 导出目标和命名空间

安装文件还不够。消费者应该获得一个带完整使用要求的 CMake 目标，而不是自己手写：

```cmake
include_directories(...)
link_directories(...)
target_link_libraries(...绝对路径...)
```

Day52 把目标导出：

```cmake
install(
    EXPORT day52Targets
    FILE day52Targets.cmake
    NAMESPACE day52::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/day52
)
```

内部目标名是：

```text
day52_pipeline
```

通过：

```cmake
set_target_properties(day52_pipeline PROPERTIES
    EXPORT_NAME pipeline
)
```

安装后对外名称变成：

```text
day52::pipeline
```

`day52::` 命名空间的好处：

- 一眼看出它是导入或别名目标；
- 降低与消费者自己的 `pipeline` 目标重名的概率；
- 若 `find_package()` 没成功，CMake 会更早发现目标不存在。

---

## 15. day52Config.cmake.in

消费者执行：

```cmake
find_package(day52 CONFIG REQUIRED)
```

时，CMake 需要找到 `day52Config.cmake`。这个文件由模板生成：

```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

find_dependency(OpenCV 5 REQUIRED COMPONENTS core imgcodecs imgproc)
find_dependency(Threads REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/day52Targets.cmake")

check_required_components(day52)
```

逐项解释：

### `@PACKAGE_INIT@`

由 `configure_package_config_file()` 展开，提供生成可迁移包配置所需的初始化逻辑和辅助宏。

### `find_dependency()`

Day52 库本身依赖 OpenCV 和 Threads。消费者导入 `day52::pipeline` 之前，必须先创建这些依赖目标。

它与普通 `find_package()` 类似，但会正确继承外层包查找的 `REQUIRED`、失败状态等语义。

### `CMAKE_CURRENT_LIST_DIR`

表示当前 `day52Config.cmake` 所在目录。这样它可以从自己的当前位置加载 `day52Targets.cmake`，而不是依赖源码绝对路径。

### `check_required_components(day52)`

检查消费者有没有请求不存在的必要组件。即使当前 Day52 没划分组件，保留它也是推荐的包配置结构。

---

## 16. 配置文件与版本文件

生成包配置：

```cmake
configure_package_config_file(
    cmake/day52Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/day52Config.cmake
    INSTALL_DESTINATION ${DAY52_CMAKE_INSTALL_DIR}
)
```

这里没有使用普通 `configure_file()`，因为 `configure_package_config_file()` 专门处理安装包的可迁移路径。

生成版本文件：

```cmake
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/day52ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
```

项目版本为 `1.0.0`。`SameMajorVersion` 表示消费者请求 1.x 版本时，可以接受同一主版本内兼容的版本。

消费者当前写的是：

```cmake
find_package(day52 1 CONFIG REQUIRED)
```

它要求找到主版本为 1 的 Day52 配置包。

---

## 17. 独立消费者工程

消费者位于：

```text
tests/consumer/
```

它是独立 CMake 工程，不通过 `add_subdirectory()` 直接访问 Day52 源码。

核心 CMake：

```cmake
find_package(day52 1 CONFIG REQUIRED)

add_executable(day52_consumer main.cpp)

target_link_libraries(day52_consumer
    PRIVATE day52::pipeline
)
```

源码只使用正常安装头路径：

```cpp
#include <day52/image_pipeline.hpp>
```

没有出现：

```text
../../include
D:/opencv-learning/...
```

消费者成功调用：

- `parse_operation("edge")`；
- `parse_execution_mode("threads")`；
- `operation_name()`；
- `resolve_worker_count()`。

运行结果：

```text
Installed target: day52::pipeline
DAY52_CONSUMER_OK
```

这证明的不是“源码碰巧能编译”，而是安装后的头文件、静态库、导出目标和依赖传播能够共同工作。

---

## 18. CPack ZIP 打包

配置：

```cmake
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "day52-cpp-image-pipeline")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-windows-ucrt64"
)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
include(CPack)
```

生成命令：

```powershell
cmake --build build --target package
```

输出：

```text
build/day52-cpp-image-pipeline-1.0.0-windows-ucrt64.zip
```

实际大小：

```text
710031 bytes，约 693.39 KiB
```

CPack 二进制包的内容来自 `install()` 规则，而不是粗暴压缩整个源码目录或构建目录。因此：

- `install()` 决定发布内容；
- CPack 决定怎样封装这些发布内容；
- 不在安装规则中的单元测试、缓存和中间文件不会混入 ZIP。

---

## 19. 包验证

`tests/day52_package_check.ps1` 检查：

- ZIP 文件存在；
- 安装树有 EXE；
- 安装树有静态库；
- 安装树有公开头文件；
- 安装树有 Config、Version 和 Targets 文件；
- ZIP 解压后同样包含必要文件；
- `day52Config.cmake` 没有硬编码 `D:\opencv-learning`；
- 导出目标名为 `day52::pipeline`。

结果：

```text
Passed 13 Day52 package checks.
DAY52_PACKAGE_OK
```

随后用解压目录作为新的前缀：

```text
build/package-check
```

重新配置了另一个消费者构建目录：

```text
build/consumer-zip
```

结果仍然输出：

```text
DAY52_CONSUMER_OK
```

最后直接运行解压包中的程序：

```text
build/package-check/bin/day52_parallel_pipeline.exe
```

实际结果：

```text
Operation: edge
Mode: threads
Requested workers: 4
Effective workers: 4
Hardware concurrency: 12
Files: 4
Elapsed: 10.796 ms
Throughput: 370.52 images/s
Saved 4 / 4 image(s); read-failed 0; write-failed 0.
DAY52_PIPELINE_OK
```

这次时间只用于证明包内程序能够运行，不是严格性能基准，不能与 Day51 的 32 图重复实验直接比较。

---

## 20. “可迁移”不等于“完全独立运行”

Day52 ZIP 是可迁移的 CMake 开发包，含义是：

- 可以移动到其他目录；
- 包配置不依赖原 Day52 源码绝对路径；
- 在兼容的编译器和依赖环境中，消费者可以用 `find_package()` 使用它。

但它不是完全独立的最终用户软件。

对包内 EXE 执行 `objdump -p`，检测到的主要非系统 DLL：

```text
libgcc_s_seh-1.dll
libwinpthread-1.dll
libstdc++-6.dll
libopencv_core-500.dll
libopencv_imgcodecs-500.dll
libopencv_imgproc-500.dll
```

当前测试能够运行，是因为：

```text
D:\VScode_C++\msys64\ucrt64\bin
```

已经加入 `PATH`，Windows 可以在那里找到 DLL。

如果把当前 ZIP 复制到一台完全没有 MSYS2/OpenCV 的电脑，EXE 很可能会因为缺少 DLL 而无法启动。要做真正独立部署，后续还需要：

1. 收集允许分发的运行时 DLL；
2. 处理 OpenCV 的间接依赖；
3. 确认许可证和再分发要求；
4. 在干净 Windows 环境或虚拟机中测试；
5. 必要时生成安装器。

因此今天应准确表述为：

> Day52 生成了经过安装树、消费者和解压验证的 CMake ZIP 开发包；它在兼容的 MSYS2 UCRT64 + OpenCV 5 环境中可迁移使用，但尚未捆绑全部运行时 DLL，不是免环境的最终用户安装包。

---

## 21. 完整复现命令

### 21.1 准备环境

```powershell
$env:MSYS2_ROOT = "D:\VScode_C++\msys64"
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"

Set-Location "D:\opencv-learning\52_cpp_cmake_install_ctest"
```

### 21.2 配置

```powershell
cmake --preset msys2-ucrt64-debug
```

### 21.3 干净构建

```powershell
cmake --build --preset msys2-ucrt64-debug --clean-first
```

### 21.4 运行全部 CTest 测试

简洁输出：

```powershell
ctest --preset msys2-ucrt64-debug
```

详细输出：

```powershell
ctest --preset msys2-ucrt64-debug -V
```

只在失败时显示测试输出：

```powershell
ctest --test-dir build --output-on-failure --no-tests=error
```

### 21.5 生成安装树

```powershell
cmake --install build
```

安装前缀已经在 Preset 中设置为：

```text
build/install
```

也可以临时覆盖：

```powershell
cmake --install build --prefix build/another-install
```

### 21.6 配置安装树消费者

```powershell
cmake -S tests/consumer -B build/consumer -G Ninja `
    "-DCMAKE_BUILD_TYPE=Debug" `
    "-DCMAKE_CXX_COMPILER=$env:MSYS2_ROOT\ucrt64\bin\g++.exe" `
    "-DCMAKE_PREFIX_PATH=$PWD\build\install" `
    "-DOpenCV_DIR=$env:MSYS2_ROOT\ucrt64\lib\cmake\opencv5"
```

`CMAKE_PREFIX_PATH` 告诉 `find_package()` 去哪个安装前缀查找包。它指向前缀根目录，不必直接写到 `lib/cmake/day52`。

### 21.7 构建并运行消费者

```powershell
cmake --build build/consumer --clean-first
.\build\consumer\day52_consumer.exe
```

### 21.8 生成 ZIP

```powershell
cmake --build build --target package
```

### 21.9 检查安装树和 ZIP

```powershell
.\tests\day52_package_check.ps1
```

---

## 22. 首次测试失败与修复

第一次运行 CTest 时：

- 21 项 C++ 单元断言全部通过；
- 契约测试前 46 项通过；
- 在准备损坏图片输入目录时失败。

错误位于：

```powershell
Copy-Item -LiteralPath (Join-Path $assets "*") ...
```

错误信息说明路径末尾的 `*` 不存在。

根因是：

- `-LiteralPath` 会把路径按字面解释；
- 星号不会被解释成通配符；
- 因此 PowerShell 真的去寻找名字为 `*` 的文件。

修复采用显式枚举文件：

```powershell
Get-ChildItem -LiteralPath $assets -File |
    Copy-Item -Destination $corruptInput -Force
```

这样既保留了 `LiteralPath` 对特殊字符的安全处理，又把“选择哪些文件”交给 `Get-ChildItem` 明确完成。

修复后重新运行完整 CTest，2/2 个测试全部通过。这个问题说明：测试脚本本身也是代码，也需要观察失败、定位根因和回归验证。

---

## 23. 今天四个关键问题

### 问题一：`cmake --build` 和 `cmake --install` 有什么区别？

`build` 负责编译，产生服务于当前构建过程的 EXE、静态库、对象文件和缓存；`install` 根据人工定义的安装规则，挑选对外文件并重组为稳定的 `bin/include/lib` 结构。

### 问题二：为什么区分 `BUILD_INTERFACE` 和 `INSTALL_INTERFACE`？

因为项目在源码树中构建时和安装后被消费时，公开头文件所在位置不同。区分两者可以避免把本机源码绝对路径泄漏到安装包，让导出目标能够随安装前缀移动。

### 问题三：四种测试分别验证什么？

- 单元测试验证函数和边界策略；
- 契约测试验证完整命令行程序和文件行为；
- 消费者测试验证安装后的库可以通过 `find_package()` 使用；
- 包测试验证安装树和 ZIP 内容完整、路径没有写死。

### 问题四：为什么有 ZIP 还不能在任意 Windows 电脑直接运行？

因为 ZIP 目前含自己的 EXE、静态库、头文件和 CMake 配置，但 EXE 仍动态依赖 OpenCV、libstdc++、libgcc 和 libwinpthread DLL。包的 CMake 路径是可迁移的，不代表运行时依赖已经全部捆绑。

---

## 24. 关键记忆点

1. 单元测试直接验证小范围逻辑，端到端测试验证完整用户流程，两者互补。
2. CTest 根据测试命令的退出码判断成功或失败。
3. `add_executable()` 构建测试，`add_test()` 才让 CTest 发现测试。
4. `include(CTest)` 提供默认开启的 `BUILD_TESTING`。
5. 现代 CMake 的核心是目标及其使用要求，不是全局路径变量。
6. `PUBLIC/PRIVATE/INTERFACE` 描述依赖怎样传播给消费者。
7. `BUILD_INTERFACE` 服务源码树，`INSTALL_INTERFACE` 服务安装树。
8. `install(TARGETS)` 安装编译目标，`install(EXPORT)` 重建可导入目标。
9. `day52::pipeline` 比裸文件路径更完整，因为它携带头文件和链接要求。
10. `configure_package_config_file()` 用于生成可迁移的包配置。
11. `find_dependency()` 先恢复导出目标依赖的外部包。
12. CPack 二进制包收集的是安装规则内容，不是整个 `build/`。
13. ZIP 可移动不等于程序已经无外部 DLL 依赖。
14. 包是否可用必须由独立消费者验证，不能只看文件是否存在。
15. 测试脚本也会有错误，修复后必须重新运行完整测试套件。

---

## 25. Day53 方向

按照当前路线，Day52 已经完成 C++/OpenCV 桌面工程阶段的“构建、测试、安装和开发包”闭环。Day53 可以进入以下方向之一：

1. ONNX Runtime C++ 推理：把训练模型接到 C++ 工程；
2. 嵌入式视觉准备：围绕 Raspberry Pi 或 Jetson 学习交叉平台部署；
3. Windows 运行时部署：自动收集 DLL，并在干净环境验证独立运行。

最终方向应优先服从导师课题和实验室硬件。如果暂时没有明确要求，ONNX Runtime C++ 推理与现有深度学习项目衔接最直接。
