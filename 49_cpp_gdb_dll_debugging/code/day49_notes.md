# Day49：GDB 调试与 Windows 运行时 DLL 分析

## 学习定位

Day44-Day48 已经完成 OpenCV C++ 工具链、图像读写、`cv::Mat` 内存语义、批处理、结果落盘、智能指针和可配置处理管线。Day49 不增加新的图像算法，重点转向两个工程问题：

1. C++ 程序行为异常时，如何停在源码位置并检查参数、局部变量和调用栈；
2. OpenCV 程序为什么在当前终端可以运行，换一个终端或设备后却可能提示缺少 DLL。

本日使用 `vector::at()` 产生受检查的越界异常，不使用数组越界、空指针解引用等可能破坏内存的未定义行为。

## 环境

- 编译器：MSYS2 UCRT64 G++ 16.2.0
- 调试器：GNU GDB 17.2
- 二进制分析：GNU objdump 2.47
- CMake：4.4.2
- Ninja：1.13.2
- OpenCV：5.0.0
- C++ 标准：C++17
- 工具链根目录：`D:\VScode_C++\msys64`
- 构建类型：`Debug`

本机没有 `ntldd.exe` 或 `ldd.exe`，因此使用工具链已有的 `objdump -p` 分析 PE 导入 DLL，不为本课安装新依赖。

## 目录结构

```text
49_cpp_gdb_dll_debugging/
├── assets/
│   └── day45_input.jpg
├── code/
│   ├── day49_gdb_dll_debugging.cpp
│   └── day49_notes.md
├── tests/
│   ├── day49_breakpoint.gdb
│   ├── day49_contract.ps1
│   ├── day49_exception.gdb
│   └── day49_shared_libraries.gdb
├── .gitignore
├── CMakeLists.txt
└── CMakePresets.json
```

`build/` 由 `.gitignore` 排除，不上传可执行文件、目标文件和 CMake 缓存。

## 1. 先区分四类问题

| 阶段 | 典型现象 | 主要检查工具 |
| --- | --- | --- |
| 编译 | 语法错误、类型不匹配、找不到头文件 | 编译器诊断、CMake 构建输出 |
| 链接 | `undefined reference`、找不到 `.a` 导入库 | 链接命令、CMake target 配置 |
| 程序加载 | Windows 提示缺少某个 DLL，`main` 尚未执行 | `objdump -p`、`where.exe`、`PATH` |
| 运行 | 参数错误、异常、崩溃、结果不符合预期 | GDB、日志、自动测试 |

GDB 主要处理运行阶段的问题。DLL 缺失发生在 Windows 加载程序准备进程时，程序甚至可能还没有进入 `main`，因此不能把所有失败都统称为“代码运行错误”。

## 2. Debug 与 Release

Day48 使用 `Release`；Day49 切换为 `Debug`：

```json
"CMAKE_BUILD_TYPE": "Debug"
```

目标编译选项：

```cmake
-O0
-g3
-Wall
-Wextra
-pedantic
```

最终 Ninja 编译参数实测为：

```text
-g -std=c++17 -O0 -g3 -Wall -Wextra -pedantic
```

- `-O0`：关闭优化，使源码执行顺序、局部变量和栈帧更容易观察；
- `-g3`：生成较完整的调试符号；
- `-Wall -Wextra -pedantic`：在运行前尽量暴露可疑写法；
- Release 优化可能合并、移动或删除代码，GDB 中可能出现 `<optimized out>`，不适合初学源码级调试。

Debug 构建不代表代码自动正确，它只是让程序内部状态更容易被观察。

## 3. 调试目标程序

命令格式：

```text
day49_debug_target <image_path> <summary_index>
```

程序读取彩色图片并建立摘要：

```cpp
const std::vector<int> summary{
    image.cols,
    image.rows,
    image.channels(),
};
```

对测试图片而言：

```text
summary = {427, 640, 3}
```

`select_summary_value` 使用受检查的访问：

```cpp
return summary.at(index);
```

索引 0、1、2 分别选择宽度、高度和通道数；索引 9 会抛出 `std::out_of_range`。程序最终捕获异常并返回确定性退出码 4，但 GDB 可以在异常产生前停住。

### 退出码

| 场景 | 退出码 |
| --- | ---: |
| 正常完成 | 0 |
| 图片无法读取 | 1 |
| 参数数量错误 | 2 |
| 索引不是非负整数 | 3 |
| 摘要索引越界 | 4 |

索引使用 `std::from_chars` 解析，能够拒绝空字符串、负数、字母和只解析了一部分的字符串，而不依赖异常完成普通输入校验。

## 4. 构建和普通运行

```powershell
cd D:\opencv-learning\49_cpp_gdb_dll_debugging
$env:MSYS2_ROOT = 'D:\VScode_C++\msys64'
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"

cmake --preset msys2-ucrt64-debug
cmake --build --preset msys2-ucrt64-debug --clean-first

.\build\day49_debug_target.exe .\assets\day45_input.jpg 1
```

实测输出：

```text
Image: 427 x 640 channels=3
summary[1]=640
DAY49_DEBUG_OK
VALID_EXIT=0
```

负路径实测：

```text
MISSING_IMAGE_EXIT=1
ARGS_EXIT=2
INVALID_INDEX_EXIT=3
RANGE_EXIT=4
```

## 5. GDB 启动、断点和源码定位

启动方式：

```powershell
gdb .\build\day49_debug_target.exe
```

常用交互命令：

```gdb
set pagination off
break main
break select_summary_value
run assets/day45_input.jpg 1
next
step
list
print index
info args
info locals
backtrace
continue
quit
```

### 命令含义

- `break function`：在函数入口设置断点；
- `run args...`：带参数启动被调试程序；
- `next`：执行下一源码行，遇到函数调用时不进入函数；
- `step`：执行下一源码行，进入有调试信息的函数；
- `list`：显示当前位置附近源码；
- `print expression`：计算并显示表达式；
- `info args`：查看当前栈帧的函数参数；
- `info locals`：查看当前栈帧的局部变量；
- `backtrace` 或 `bt`：从当前位置向调用者打印调用链；
- `continue`：继续运行到下一个断点或程序结束。

自动断点脚本：

```powershell
gdb -q -batch `
    -x .\tests\day49_breakpoint.gdb `
    .\build\day49_debug_target.exe
```

实测停在：

```text
Breakpoint 1, main (...) at day49_gdb_dll_debugging.cpp:33
Breakpoint 2, select_summary_value (..., index=1) at day49_gdb_dll_debugging.cpp:29
```

GDB 能看到：

```text
summary = std::vector of length 3, capacity 3 = {427, 640, 3}
index = 1
```

调用栈：

```text
#0 select_summary_value(..., index=1) at day49_gdb_dll_debugging.cpp:29
#1 main(...) at day49_gdb_dll_debugging.cpp:58
```

`#0` 是当前函数，`#1` 是调用它的 `main`。数字越大，位置越靠近调用链上游。

## 6. 调试 C++ 越界异常

### 通用方法与本机差异

GDB 通用文档通常建议：

```gdb
catch throw
run assets/day45_input.jpg 9
backtrace
```

但 C++ 异常 catchpoint 依赖平台、ABI 和 `libstdc++` 是否提供相应探针。本机 MSYS2 UCRT64/GDB 17.2 的真实结果是：

```text
Catchpoint 1 (throw)
Summary index out of range: 9
Inferior exited with code 04
No stack.
```

也就是说，catchpoint 创建成功，但抛出 `std::out_of_range` 时没有命中。继续执行 `backtrace` 时程序已经退出，所以只能得到 `No stack`。

### 本机可靠方法

通过 PE 导入表和符号检查，程序实际调用的是：

```text
std::__throw_out_of_range_fmt(char const*, ...)
```

因此异常脚本使用：

```gdb
break 'std::__throw_out_of_range_fmt(char const*, ...)'
run assets/day45_input.jpg 9
backtrace
frame 3
info args
info locals
continue
```

实测调用栈：

```text
#0 std::__throw_out_of_range_fmt(...)
#1 std::vector<int>::_M_range_check(..., __n=9)
#2 std::vector<int>::at(..., __n=9)
#3 select_summary_value(..., index=9) at day49_gdb_dll_debugging.cpp:29
#4 main(...) at day49_gdb_dll_debugging.cpp:58
```

调用链应从下向上理解：

```text
main
→ select_summary_value
→ vector::at
→ _M_range_check
→ __throw_out_of_range_fmt
```

使用 `frame 3` 回到自己的代码后，GDB 显示：

```text
summary = std::vector of length 3, capacity 3 = {427, 640, 3}
index = 9
```

根因因此很明确：合法索引只有 0、1、2，调用者传入了 9。调试时不要只停在标准库最深处，应使用 `backtrace` 找到第一个属于自己代码的栈帧。

注意：`std::__throw_out_of_range_fmt` 是当前 GCC/libstdc++ 实现细节，不是跨工具链稳定的标准 C++ API。换成 MSVC、Clang 或其他 libstdc++ 版本时，应重新检查实际异常入口，优先尝试平台支持的 `catch throw`。

## 7. 使用 `objdump` 查看静态 DLL 依赖

Windows 可执行文件采用 PE 格式。下面的命令查看 exe 导入表中声明的 DLL：

```powershell
& "$env:MSYS2_ROOT\ucrt64\bin\objdump.exe" `
    -p .\build\day49_debug_target.exe |
    Select-String 'DLL Name'
```

与本课最相关的直接导入项：

```text
DLL Name: libgcc_s_seh-1.dll
DLL Name: libstdc++-6.dll
DLL Name: libopencv_core-500.dll
DLL Name: libopencv_imgcodecs-500.dll
```

- `libgcc_s_seh-1.dll`：GCC 异常处理等运行时支持；
- `libstdc++-6.dll`：GNU C++ 标准库；
- `libopencv_core-500.dll`：OpenCV 核心数据结构；
- `libopencv_imgcodecs-500.dll`：图像编解码模块。

`objdump -p` 看到的是当前 exe 的 PE 导入项，不一定列出所有传递依赖。例如 `libopencv_imgcodecs-500.dll` 自己还会依赖 JPEG、PNG、TIFF、WebP 等编解码 DLL。

## 8. 使用 GDB 查看运行时实际加载的 DLL

```gdb
start assets/day45_input.jpg 1
info sharedlibrary
continue
```

实测 GDB 不仅看到直接依赖，还看到运行时实际加载的传递依赖，包括：

```text
libopencv_core-500.dll
libopencv_imgcodecs-500.dll
libopencv_imgproc-500.dll
libstdc++-6.dll
libgcc_s_seh-1.dll
libjpeg-8.dll
libpng16-16.dll
libtiff-6.dll
libwebp-7.dll
zlib1.dll
```

因此：

```text
objdump -p：静态查看 exe 直接声明的导入项
info sharedlibrary：查看当前运行真正加载到进程中的 DLL
```

两者回答的问题不同，应该结合使用。

## 9. Windows `PATH` 与 DLL 搜索

开发终端中先加入 UCRT64：

```powershell
$env:MSYS2_ROOT = 'D:\VScode_C++\msys64'
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"
```

然后查询：

```powershell
where.exe libopencv_core-500.dll
where.exe libopencv_imgcodecs-500.dll
where.exe libstdc++-6.dll
```

实测位置：

```text
D:\VScode_C++\msys64\ucrt64\bin\libopencv_core-500.dll
D:\VScode_C++\msys64\ucrt64\bin\libopencv_imgcodecs-500.dll
D:\VScode_C++\msys64\ucrt64\bin\libstdc++-6.dll
```

把 `PATH` 临时缩减为 Windows 系统目录后：

```text
CLEAN_PATH_WHERE_EXIT=1
```

这表示 Windows 搜索路径中已经找不到 OpenCV DLL。程序在当前开发终端能运行，依赖的是 UCRT64 `bin` 已位于 `PATH` 前部，而不是 exe 已经包含了 OpenCV。

本机还存在：

```text
D:\VScode_C++\mingw64\bin\libstdc++-6.dll
```

同名 DLL 同时存在于 `ucrt64` 和 `mingw64` 时，必须保证所用编译器、OpenCV 和运行库来自同一套环境。本课由 UCRT64 编译，因此把 `ucrt64\bin` 放在前面，不能随意混用两套 ABI。

开发阶段可使用受控 `PATH`；真正交付到其他电脑或嵌入式设备时，需要单独设计依赖打包、安装位置或静态链接策略，不能简单假设目标机器也安装了同一套 MSYS2。

## 10. 自动验收

运行：

```powershell
cd D:\opencv-learning\49_cpp_gdb_dll_debugging
$env:MSYS2_ROOT = 'D:\VScode_C++\msys64'
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"
.\tests\day49_contract.ps1
```

契约测试检查：

1. Debug 可执行文件存在；
2. GDB 和 objdump 可用；
3. 正常输入的尺寸、通道、摘要值、成功标记和退出码；
4. 图片缺失、参数缺失、索引格式错误和索引越界退出码；
5. GDB 能停在 `main` 和 `select_summary_value`；
6. GDB 能显示 `index=1`、Day49 源文件和调用栈；
7. 异常断点能回溯到 `select_summary_value(index=9)`；
8. PE 导入表包含 OpenCV 和 MinGW C++ 运行库；
9. GDB 能看到运行时加载的 OpenCV DLL。

最终实测 24 项全部通过：

```text
DAY49_CONTRACT_OK
TEST_EXIT=0
```

## 遇到的问题与解决方法

### 问题：`catch throw` 创建成功但没有命中

现象：程序返回退出码 4 后，GDB 报 `No stack`。

调查过程：

1. 单独运行异常脚本，确认失败可以稳定复现；
2. 尝试在通用 ABI 入口 `__cxa_throw` 设置 pending breakpoint，仍未命中；
3. 使用 `objdump` 和 `nm` 检查实际导入符号；
4. 发现程序导入 `std::__throw_out_of_range_fmt`；
5. 在该符号设置断点后，调用栈稳定回到 Day49 第 29 行并显示 `index=9`。

根因：当前 MinGW/libstdc++ 构建没有提供让 GDB `catch throw` 正常工作的异常探针或可见 ABI 入口，通用方法在本机不适用。

修正只调整 GDB 调试脚本，没有改变程序的业务行为。契约测试随后全部通过。

## 关键记忆点

1. Debug 构建的 `-O0 -g3` 让源码、变量和栈帧更容易观察。
2. `break` 控制停止位置，`next` 跨过函数，`step` 进入函数。
3. `info args`、`info locals` 和 `print` 查看当前栈帧状态。
4. `backtrace` 先展示完整调用链，再切换到第一个属于自己代码的栈帧。
5. `catch throw` 的可用性依赖平台；工具命令创建成功不代表运行时一定能命中。
6. `vector::at()` 会进行范围检查，适合构造安全、确定性的调试练习。
7. `objdump -p` 查看 exe 的直接 PE 导入，`info sharedlibrary` 查看运行时实际加载的 DLL。
8. 链接成功不代表程序能在任意终端运行，Windows 加载器仍需找到所有 DLL。
9. UCRT64 和 MINGW64 中可能存在同名 DLL，不能混用编译器、OpenCV 和运行库。
10. 调试失败本身也要记录；真实限制比只保留最终成功命令更有学习价值。

## 参考资料

- GNU GDB 官方手册：<https://sourceware.org/gdb/current/onlinedocs/gdb>
- GNU objdump 官方文档：<https://sourceware.org/binutils/docs/binutils/objdump.html>

## Day50 计划

把目前集中在单个 `.cpp` 文件中的处理逻辑重构为类和多个源文件，学习头文件与实现文件分离、构造函数、接口职责、CMake 多源文件 target，以及小型 C++ OpenCV 工程的目录边界。
