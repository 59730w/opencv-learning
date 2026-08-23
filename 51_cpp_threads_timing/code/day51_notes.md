# Day51：C++ 多线程批处理、数据竞争与性能计时

日期：2026-08-23

学习阶段：C++ OpenCV 工程化过渡

前置内容：

- Day48：可配置 `gray / blur / edge` 图像处理管线；
- Day49：Debug 构建、GDB 和 DLL 依赖；
- Day50：`ImagePipeline` 类、头源分离、静态库与可执行程序目标。

阅读时建议同时打开以下三个文件：

- `include/day51/image_pipeline.hpp`
- `src/image_pipeline.cpp`
- `code/day51_threads_timing.cpp`

---

## 1. 今天解决什么问题

Day50 的批处理管线已经具有清晰的类边界，但处理图片时仍然逐张执行：

```text
读取图片1 → 处理图片1 → 写出图片1
                            ↓
读取图片2 → 处理图片2 → 写出图片2
                            ↓
读取图片3 → 处理图片3 → 写出图片3
```

如果每张图片之间互不依赖，可以让多个 CPU 工作线程同时处理不同图片：

```text
线程1：图片1 → 图片5 → 图片9  → ...
线程2：图片2 → 图片6 → 图片10 → ...
线程3：图片3 → 图片7 → 图片11 → ...
线程4：图片4 → 图片8 → 图片12 → ...
```

但“把循环放进多个线程”不是完整答案。并行程序首先必须回答：

1. 如何保证两个线程不会领取同一张图片？
2. 如何避免多个线程同时修改 `std::vector` 的内部结构？
3. 如何保证输出顺序稳定？
4. 工作线程发生异常时怎样避免程序直接终止？
5. 主线程怎样知道所有工作线程已经完成？
6. 怎样证明并行结果与顺序结果一致？
7. 怎样公平测量性能，而不是只挑最快的一次？
8. OpenCV 自己的内部线程会不会干扰本次比较？

Day51 的核心顺序是：

```text
正确性 → 线程安全 → 可重复计时 → 性能解释
```

速度排在最后。

---

## 2. 今日完成内容

本节完成了：

1. `sequential` 顺序执行模式；
2. `threads` 标准库线程模式；
3. `std::thread` 创建工作线程；
4. Lambda 引用捕获共享上下文；
5. `std::atomic<std::size_t>` 分发任务下标；
6. `fetch_add(1)` 保证每个下标只被领取一次；
7. 预分配结果向量并按唯一位置写入；
8. `joinable()` 与 `join()` 回收线程；
9. `std::exception_ptr` 保存线程异常；
10. `std::mutex` 保护第一份异常；
11. `std::chrono::steady_clock` 测量处理时间；
12. 计算端到端图片吞吐量；
13. 使用 `cv::setNumThreads(1)` 隔离 OpenCV 内部并行；
14. 自动生成32张临时基准图片；
15. 顺序与1/2/4线程输出数量和 SHA256 一致性检查；
16. 每组预热一次、正式测量三次并取中位数；
17. 71项自动契约检查；
18. 顺序与4线程结果的实际视觉检查；
19. 对“线程越多不一定越快”进行了实测解释。

---

## 3. 项目目录结构

```text
51_cpp_threads_timing/
├── .gitignore
├── CMakeLists.txt
├── CMakePresets.json
├── assets/
│   └── input/
│       └── day45_input.jpg
├── include/
│   └── day51/
│       └── image_pipeline.hpp
├── src/
│   └── image_pipeline.cpp
├── code/
│   ├── day51_threads_timing.cpp
│   └── day51_notes.md
└── tests/
    ├── prepare_benchmark.ps1
    ├── day51_contract.ps1
    └── day51_benchmark.ps1
```

生成物全部位于：

```text
build/
```

其中包括：

- 静态库；
- 可执行程序；
- 32张临时输入图片；
- 顺序和多线程输出；
- 基准测试原始 CSV；
- 基准测试汇总 CSV。

这些内容可重新生成，因此由 `.gitignore` 忽略。

---

## 4. 进程与线程

### 4.1 进程

进程可以理解为一个正在运行的程序实例。它拥有自己的地址空间和操作系统资源。

例如运行：

```powershell
.\build\day51_parallel_pipeline.exe ...
```

操作系统会创建一个新的进程。

### 4.2 线程

线程是进程内部的一条执行路径。同一进程中的线程通常共享：

- 代码；
- 全局数据；
- 堆内存；
- 文件句柄；
- 进程地址空间。

每个线程独立拥有：

- 调用栈；
- 当前执行位置；
- 部分线程局部状态。

### 4.3 共享内存既是优势也是风险

共享内存让线程之间交换数据很方便，但也意味着两个线程可能同时访问同一个对象。

只读共享通常容易保证安全：

```cpp
const std::vector<fs::path>& images
```

多个线程都只读 `images`，不会修改它的大小或元素，因此这部分没有写入冲突。

如果多个线程同时修改同一个非原子对象，而且缺少同步，就可能产生数据竞争。

---

## 5. 什么是数据竞争

当以下条件同时成立时，会产生数据竞争：

1. 两个或更多线程访问同一个内存位置；
2. 至少一个线程在写；
3. 访问之间没有正确同步。

数据竞争会导致未定义行为。未定义行为不只是“结果偶尔不正确”，还可能表现为：

- 程序崩溃；
- 输出随机变化；
- Debug 正常但 Release 出错；
- 换一台电脑后出错；
- 增加日志后问题暂时消失；
- 编译器优化后出现完全不同的结果。

因此不能用“我运行一次没崩”证明线程安全。

---

## 6. 一个错误的并行写法

下面的想法看起来直接，但不安全：

```cpp
std::vector<ImageResult> results;

// 多个线程同时执行
results.push_back(process_image(...));
```

`push_back()` 可能修改：

- `vector` 当前大小；
- 容量；
- 内部指针；
- 元素存储位置。

容量不足时还可能重新分配整块内存。多个线程同时 `push_back()` 会争用同一个容器内部状态。

即使提前调用 `reserve()`，也不能让多个线程同时修改 `size()`。`reserve()` 只减少重新分配，不会把并发 `push_back()` 自动变成安全操作。

---

## 7. Day51 的安全结果写入设计

### 7.1 先固定结果数量

图片列表确定后，先执行：

```cpp
report.results.resize(images.size());
```

此时结果向量已经拥有固定数量的元素，线程运行期间不再改变容器大小。

### 7.2 每个线程只写唯一下标

```cpp
results[index] = process_image(images[index], output_dir);
```

只要不同线程得到的 `index` 不同，它们就在写不同的 `ImageResult` 对象。

Day51 不允许：

- 在线程中 `push_back()`；
- 在线程中 `resize()`；
- 在线程中清空结果向量；
- 两个线程处理同一索引。

### 7.3 输出顺序保持稳定

输入图片先按路径排序：

```cpp
std::sort(images.begin(), images.end());
```

无论哪个线程先完成，结果始终写回原始下标：

```text
images[0] → results[0]
images[1] → results[1]
images[2] → results[2]
```

因此完成顺序可能变化，但最终结果顺序稳定。

---

## 8. `std::atomic` 原子任务索引

### 8.1 普通整数为什么不够

错误示例：

```cpp
std::size_t next_index = 0;
const std::size_t index = next_index++;
```

`next_index++` 不是不可分割的一步。它通常包含：

1. 读取旧值；
2. 计算旧值加一；
3. 写回新值。

两个线程可能同时读到同一个旧值，于是都领取到同一张图片。

### 8.2 原子变量

Day51 使用：

```cpp
std::atomic<std::size_t> next_index{0};
```

领取任务：

```cpp
const std::size_t index = next_index.fetch_add(1);
```

`fetch_add(1)` 会原子地：

1. 返回当前值；
2. 把内部值增加一。

如果初始值是0，四个线程可能分别得到：

```text
线程A → 0
线程B → 1
线程C → 2
线程D → 3
```

具体哪个线程拿到哪个数字不确定，但同一个数字不会被成功领取两次。

### 8.3 动态任务分配

每处理完一张图，线程继续领取下一个下标：

```cpp
while (!stop_requested.load()) {
    const std::size_t index = next_index.fetch_add(1);
    if (index >= images.size()) {
        break;
    }
    results[index] = process_image(images[index], output_dir);
}
```

这种方式比固定给每个线程分配相同数量图片更灵活。如果某张图片处理较慢，先完成的线程可以继续领取后面的任务。

### 8.4 为什么本节不讲内存序

`std::atomic` 允许指定更细的内存顺序，例如 `memory_order_relaxed`。Day51 使用默认顺序，因为：

- 默认行为更容易正确理解；
- 32张图的性能瓶颈不在这一次原子递增；
- 过早优化内存序会增加推理难度；
- 本节重点是建立正确的线程安全模型。

---

## 9. `std::thread` 的创建

工作线程保存在：

```cpp
std::vector<std::thread> workers;
workers.reserve(effective_workers);
```

创建线程：

```cpp
workers.emplace_back([&]() {
    // 工作线程逻辑
});
```

`std::thread` 构造完成后，新线程就可能立即开始运行。

不能假设所有线程会按创建顺序启动，也不能假设线程1总是比线程2先完成。

---

## 10. Lambda 的 `[&]` 引用捕获

```cpp
[&]() {
    // 使用外层局部变量
}
```

`[&]` 表示 Lambda 中使用的外层变量按引用捕获。

本节工作线程需要访问：

- `images`；
- `output_dir`；
- `results`；
- `next_index`；
- `stop_requested`；
- 异常状态。

使用引用捕获的前提是这些对象在线程执行期间仍然存活。

Day51 中所有线程都在 `process_threads()` 返回前执行 `join()`，因此这些局部变量在线程结束前不会被销毁。

如果使用 `detach()` 让线程在函数返回后继续执行，那么引用捕获可能指向已经销毁的局部变量，产生悬空引用。

---

## 11. `joinable()` 与 `join()`

### 11.1 `join()` 的作用

```cpp
worker.join();
```

调用 `join()` 的线程会等待目标工作线程结束。

Day51 的主处理线程必须等待所有工作线程完成，才能：

- 读取全部 `results`；
- 结束计时；
- 汇总成功与失败；
- 返回 `PipelineReport`；
- 退出程序。

### 11.2 为什么先检查 `joinable()`

```cpp
if (worker.joinable()) {
    worker.join();
}
```

一个线程对象只有在关联着可等待的执行线程时才是 joinable。对不可 join 的线程调用 `join()` 会抛出异常。

### 11.3 为什么不用 `detach()`

`detach()` 会让工作线程与 `std::thread` 对象分离。分离后，当前流程无法再 `join()` 它。

本节若使用 `detach()`，可能发生：

```text
工作线程仍在写图片
        ↓
process_threads() 已返回
        ↓
results 或路径对象被销毁
        ↓
线程继续访问失效对象
```

批处理需要明确的完成点，所以使用 `join()`，不用 `detach()`。

### 11.4 忘记 join 的后果

一个仍然 joinable 的 `std::thread` 对象在析构时会调用 `std::terminate()`。所以线程回收不是可选清理步骤，而是正确性要求。

---

## 12. 工作线程中的异常问题

如果线程函数中的异常没有在该线程内捕获，它不会自动传播到创建线程的一方，而会导致 `std::terminate()`。

因此不能简单写成：

```cpp
workers.emplace_back([&]() {
    results[index] = process_image(...);  // 可能抛异常
});
```

Day51 使用：

```cpp
try {
    // 线程工作
} catch (...) {
    // 保存异常
}
```

### 12.1 `std::exception_ptr`

```cpp
std::exception_ptr first_error;
```

捕获当前异常：

```cpp
first_error = std::current_exception();
```

全部线程 join 后，在调用线程重新抛出：

```cpp
if (first_error) {
    std::rethrow_exception(first_error);
}
```

这样异常重新回到普通调用链，由 `main()` 的 `catch` 处理。

### 12.2 为什么需要互斥锁

可能有多个线程几乎同时失败。它们不能同时写同一个 `first_error`。

```cpp
std::mutex error_mutex;
```

写入时：

```cpp
const std::lock_guard<std::mutex> lock(error_mutex);
if (!first_error) {
    first_error = std::current_exception();
}
```

`lock_guard` 创建时加锁，离开作用域时自动解锁，符合 RAII。

### 12.3 停止标记

```cpp
std::atomic<bool> stop_requested{false};
```

某个线程失败后：

```cpp
stop_requested.store(true);
```

其他线程下一次检查时停止领取新任务。已经开始的单张处理不会被强行中断。

---

## 13. 为什么工作线程不直接打印日志

虽然多个线程可以向 `std::cout` 写入，但不同线程的文本可能交错：

```text
thread1: sa thread2: saved imgved image1age2
```

交错日志很难阅读，也可能给计时增加锁竞争和 I/O 开销。

Day51 的做法是：

1. 工作线程只填充 `ImageResult`；
2. 所有线程 join；
3. 主线程按稳定顺序统一输出汇总。

这也让自动测试更稳定。

---

## 14. 顺序模式与线程模式

### 14.1 顺序模式

```cpp
for (std::size_t index = 0; index < images.size(); ++index) {
    results[index] = process_image(images[index], output_dir);
}
```

特点：

- 逻辑直接；
- 没有线程创建开销；
- 是正确性与性能基线；
- `effective_workers` 固定为1。

### 14.2 线程模式

```text
预分配 results
       ↓
创建 atomic next_index
       ↓
创建 N 个 worker
       ↓
每个 worker 动态领取 index
       ↓
写 results[index]
       ↓
join 所有线程
       ↓
重新抛出可能的 worker 异常
```

### 14.3 `threads --workers 1` 的意义

1个工作线程与顺序模式并不完全相同。

顺序模式直接在当前线程执行循环；`threads --workers 1` 仍然需要：

- 创建一个新线程；
- 使用原子变量领取任务；
- join 工作线程；
- 管理异常同步结构。

因此1线程包装模式通常可能略慢。它的价值是验证“线程框架本身”的开销和正确性。

---

## 15. 请求线程数与有效线程数

命令行中的 `--workers` 是请求值：

```text
Requested workers
```

实际创建数量还会受到图片数量限制：

```cpp
min(worker_count_, image_count)
```

例如：

```text
请求线程：8
图片数量：1
有效线程：1
```

本节测试验证了该行为。

空目录也报告有效线程1，但不会实际创建工作线程，因为 `process_threads()` 遇到空图片列表会立即返回。

### 15.1 `hardware_concurrency()`

程序还打印：

```text
Hardware concurrency: 12
```

本机当前环境报告12个硬件线程上下文。

这个值只是估计和参考：

- 不是性能保证；
- 不表示使用12线程一定最快；
- 不表示程序独占全部CPU；
- 也不等同于物理核心数量。

---

## 16. 为什么线程越多不一定越快

线程会引入额外成本：

- 创建与销毁线程；
- 原子操作；
- 操作系统调度；
- 缓存竞争；
- 内存带宽竞争；
- 磁盘读写竞争；
- JPEG 编解码开销；
- 同步和异常状态管理。

如果每个任务非常小，线程管理成本可能超过并行节省的时间。

如果任务主要受磁盘限制，增加CPU线程也不会线性加速磁盘。

如果CPU已经满载，继续增加线程只会让操作系统更频繁地切换上下文。

---

## 17. OpenCV 内部线程是一个混杂因素

许多 OpenCV 算子可能使用其自己的并行后端。如果外层再创建多个 `std::thread`，可能出现嵌套并行：

```text
4个外层线程
   ×
每个 OpenCV 调用又尝试使用多个内部线程
```

这会使基准难以解释，也可能过度订阅CPU。

Day51 在进入处理前调用：

```cpp
cv::setNumThreads(1);
```

目的不是宣称“OpenCV 永远应该单线程”，而是控制实验变量：

```text
固定 OpenCV 内部线程 = 1
只改变外层 worker 数量
```

这样顺序、1线程、2线程、4线程之间更接近对我们编写的外层并行进行比较。

### 17.1 这项设置的限制

`cv::setNumThreads()` 修改的是进程级 OpenCV 线程配置。在复杂应用中，如果多个独立模块同时改变它，可能互相影响。

Day51 是单管线教学程序，在进入并行区域前调用一次，范围可控。未来真实项目应集中管理全局并行策略，避免不同模块各自修改。

---

## 18. 计时为什么使用 `steady_clock`

系统墙上时间可能受到：

- 手动调整；
- 网络对时；
- 时区变化；
- 夏令时变化。

性能测量只关心两个时刻之间经过了多久，因此使用单调时钟：

```cpp
std::chrono::steady_clock
```

开始：

```cpp
const auto start = std::chrono::steady_clock::now();
```

结束：

```cpp
const auto end = std::chrono::steady_clock::now();
```

转换为毫秒：

```cpp
const double elapsed_ms =
    std::chrono::duration<double, std::milli>(end - start).count();
```

`duration<double, std::milli>` 表示以毫秒为单位、使用 `double` 保存的时间长度。

---

## 19. 本节的计时边界

图片列表扫描和排序发生在计时开始前。

计时包含：

```text
cv::imread
    +
gray / blur / edge
    +
cv::imwrite
```

因此本节测量的是端到端批处理耗时，而不是纯 `cv::Canny` 计算时间。

优点：

- 接近用户实际等待时间；
- 能观察完整管线吞吐量。

限制：

- 磁盘缓存会影响结果；
- JPEG 编解码占用时间；
- 不同存储设备结果不可直接比较；
- 无法单独说明算法计算加速比。

如果要测纯算法，应先把图片读入内存，再只对处理函数计时。那是另一个实验，不应与当前端到端数字混在一起。

---

## 20. 吞吐量

吞吐量表示每秒处理多少张图片：

```text
images_per_second = image_count / elapsed_seconds
```

代码等价于：

```cpp
static_cast<double>(report.results.size())
    / (report.elapsed_ms / 1000.0)
```

空目录或耗时为零时，程序输出：

```text
Throughput: 0.00 images/s
```

避免除零。

---

## 21. 命令行接口

统一格式：

```text
day51_parallel_pipeline
    <input_dir>
    <output_dir>
    --op <gray|blur|edge>
    --mode <sequential|threads>
    --workers <positive_integer>
```

### 21.1 顺序模式

```powershell
.\build\day51_parallel_pipeline.exe `
  .\build\benchmark-input `
  .\build\output-sequential `
  --op edge `
  --mode sequential `
  --workers 1
```

### 21.2 两线程模式

```powershell
.\build\day51_parallel_pipeline.exe `
  .\build\benchmark-input `
  .\build\output-threads-2 `
  --op edge `
  --mode threads `
  --workers 2
```

### 21.3 四线程模式

```powershell
.\build\day51_parallel_pipeline.exe `
  .\build\benchmark-input `
  .\build\output-threads-4 `
  --op edge `
  --mode threads `
  --workers 4
```

---

## 22. 线程数解析

主程序使用 `std::from_chars` 解析正整数：

```cpp
const auto [position, error] = std::from_chars(begin, end, value);
```

以下输入都无效：

```text
0
-1
abc
4abc
```

与 `std::stoi` 相比，`from_chars`：

- 不依赖区域设置；
- 不使用异常表示普通解析失败；
- 能精确检查是否完整消费字符串。

---

## 23. 退出码

| 退出码 | 含义 |
| ---: | --- |
| `0` | 正常完成 |
| `1` | 输入目录无效 |
| `2` | 命令行格式错误 |
| `3` | 输出目录无效或无法创建 |
| `4` | 不支持的图像操作 |
| `5` | 不支持的执行模式 |
| `6` | 工作线程数无效 |
| `7` | 处理过程中发生异常 |

成功标记：

```text
DAY51_THREADS_OK
```

退出码7主要用于捕获：

- 文件系统异常；
- OpenCV 异常；
- 工作线程保存后重新抛出的异常；
- 其他标准异常。

---

## 24. CMake 线程目标

CMake 中使用：

```cmake
find_package(Threads REQUIRED)
```

然后链接：

```cmake
target_link_libraries(day51_pipeline
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgcodecs
        opencv_imgproc
        Threads::Threads
)
```

使用 `Threads::Threads` 比手工写平台参数更可移植。

在不同平台上，它可能对应：

- 编译器线程选项；
- pthread 库；
- 平台原生线程支持；
- 不需要额外字符串但仍提供统一目标。

本机配置输出：

```text
Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
Found Threads: TRUE
```

---

## 25. 构建命令

设置环境：

```powershell
$env:MSYS2_ROOT = "D:\VScode_C++\msys64"
$env:Path = "$env:MSYS2_ROOT\ucrt64\bin;$env:Path"
```

进入课程目录：

```powershell
Set-Location D:\opencv-learning\51_cpp_threads_timing
```

配置：

```powershell
cmake --preset msys2-ucrt64-debug
```

干净构建：

```powershell
cmake --build --preset msys2-ucrt64-debug --clean-first
```

生成：

```text
build/libday51_pipeline.a
build/day51_parallel_pipeline.exe
```

编译警告选项：

```text
-Wall -Wextra -Wpedantic
```

---

## 26. 自动生成基准输入

仓库只保存一张小型测试图片，避免上传32份重复数据。

运行：

```powershell
.\tests\prepare_benchmark.ps1 -Count 32
```

脚本在：

```text
build/benchmark-input/
```

生成：

```text
tree_01.jpg
tree_02.jpg
...
tree_32.jpg
```

成功标记：

```text
DAY51_BENCHMARK_INPUT_OK
```

这些图片内容相同，因此该基准只用于控制变量和观察批处理调度，不代表真实数据分布。

---

## 27. 为什么先比较结果，再比较速度

错误的并行程序也可能非常快。

例如：

- 漏处理一半图片；
- 多个线程覆盖同一个输出文件；
- 某些线程提前退出；
- 结果写到了错误下标；
- 输出文件损坏。

如果只看耗时，这些错误甚至会让程序“看起来更快”。

Day51 正确性门槛：

```text
顺序输出数量 = 32
线程输出数量 = 32
文件名逐一匹配
SHA256逐一匹配
```

测试结果：

```text
threads-1 hashes match sequential output
threads-2 hashes match sequential output
threads-4 hashes match sequential output
```

全部通过后才进行性能解释。

---

## 28. SHA256 一致性意味着什么

SHA256 相同说明对应输出文件的字节完全一致。

它比“肉眼看起来一样”更严格，能够发现：

- 像素差异；
- 编码差异；
- 文件截断；
- 错误覆盖。

但哈希一致性仍不能单独证明实现没有任何数据竞争。它证明的是本次运行的可观察输出一致。代码设计、线程同步、重复测试和工具检查仍然重要。

---

## 29. 视觉检查

本节查看了：

- 顺序模式 `tree_01_edge.jpg`；
- 四线程模式 `tree_01_edge.jpg`。

两张图视觉表现一致：

- 黑色背景；
- 树干和树枝边缘清晰；
- 地平线与草地纹理被检测；
- 没有撕裂、空白、局部丢失或尺寸异常。

视觉检查与 SHA256 一致性相互补充。

---

## 30. 测试驱动过程

### 30.1 第一次 RED

先创建契约测试，再运行：

```text
[FAIL] public header exists
```

这证明测试能够检测 Day51 实现缺失。

### 30.2 第一次 GREEN

实现头文件、线程管线、主程序和 CMake 后：

```text
Passed 70 Day51 checks.
DAY51_CONTRACT_OK
```

### 30.3 第二次 RED：发现基准混杂因素

首次性能测试后复核发现，OpenCV 算子自身可能使用内部线程。于是先增加新测试，要求外层线程实验固定 OpenCV 内部线程为1。

新测试按预期失败：

```text
[FAIL] benchmark isolates outer threading from OpenCV internal threads
```

### 30.4 第二次 GREEN

加入：

```cpp
cv::setNumThreads(1);
```

重建后：

```text
Passed 71 Day51 checks.
DAY51_CONTRACT_OK
```

这个过程说明：测试不只是验证最终功能，也可以固定实验设计中的重要控制条件。

---

## 31. 自动契约测试范围

最终共71项检查，主要覆盖：

### 31.1 结构与实现边界

- 头文件、实现、主程序和 CMake 存在；
- 静态库和可执行程序存在；
- 声明 `ExecutionMode`；
- 声明 `PipelineReport`；
- 实现使用 `std::thread`；
- 实现使用 `std::atomic`；
- 使用 `fetch_add()`；
- 调用 `join()`；
- 使用 `steady_clock`；
- 固定 OpenCV 内部线程为1；
- 结果向量预先定长；
- `main()` 不含线程细节；
- CMake 查找并链接线程目标。

### 31.2 正常路径

- 顺序模式；
- 线程模式1个worker；
- 线程模式2个worker；
- 线程模式4个worker；
- 每组都处理32张图片；
- 每组输出模式、线程数、耗时和吞吐量；
- 每组打印成功标记；
- 所有并行输出与顺序输出哈希一致。

### 31.3 边界与错误路径

- 8个请求线程处理1张图时缩减为1；
- 空目录不除零；
- 输入目录不存在返回1；
- 参数缺失返回2；
- 输出路径是文件返回3；
- 未知图像操作返回4；
- 未知执行模式返回5；
- 线程数 `0`、`-1`、`abc` 返回6。

---

## 32. 基准测试方法

测试组合：

| 模式 | 请求线程 |
| --- | ---: |
| sequential | 1 |
| threads | 1 |
| threads | 2 |
| threads | 4 |

每个组合执行：

1. 一次预热，不记录；
2. 三次正式运行；
3. 保存三次原始耗时；
4. 使用耗时中位数；
5. 使用吞吐量中位数；
6. 计算相对顺序模式的加速比。

运行：

```powershell
.\tests\day51_benchmark.ps1 -MeasuredRuns 3
```

生成：

```text
build/day51_benchmark_raw.csv
build/day51_benchmark_summary.csv
```

---

## 33. 为什么要预热

第一次运行可能受到额外开销影响：

- DLL和代码页首次加载；
- 文件系统缓存尚未建立；
- OpenCV内部初始化；
- CPU频率状态变化；
- 输出目录首次创建。

预热不能消除所有噪声，但可以减少“第一次运行特有开销”对正式记录的影响。

预热结果不参与统计。

---

## 34. 为什么运行三次并取中位数

单次时间可能被以下因素扰动：

- 操作系统调度；
- 后台程序；
- 防病毒扫描；
- 磁盘缓存；
- CPU温度和频率；
- 其他进程的CPU或磁盘占用。

中位数是排序后位于中间的值。

例如：

```text
50 ms, 52 ms, 120 ms
```

中位数是52 ms，不会被一次120 ms的异常抖动严重拉高。

不能只记录最快一次，因为最快值可能代表最有利的偶然状态，而不是典型表现。

---

## 35. 最终原始性能数据

测试条件：

- 日期：2026-08-23；
- 输入：32张 `427 × 640` JPEG副本；
- 操作：Canny edge；
- OpenCV内部线程：1；
- 测量范围：读取 + 处理 + JPEG写入；
- 每组：1次预热 + 3次正式运行；
- 本机 `hardware_concurrency()`：12。

### 35.1 顺序模式

| 运行 | 耗时 ms | 吞吐量 images/s |
| ---: | ---: | ---: |
| 1 | 128.180 | 249.65 |
| 2 | 123.279 | 259.57 |
| 3 | 123.171 | 259.80 |

### 35.2 线程模式，1 worker

| 运行 | 耗时 ms | 吞吐量 images/s |
| ---: | ---: | ---: |
| 1 | 125.963 | 254.04 |
| 2 | 125.592 | 254.79 |
| 3 | 118.673 | 269.65 |

### 35.3 线程模式，2 workers

| 运行 | 耗时 ms | 吞吐量 images/s |
| ---: | ---: | ---: |
| 1 | 77.343 | 413.74 |
| 2 | 79.611 | 401.96 |
| 3 | 81.092 | 394.61 |

### 35.4 线程模式，4 workers

| 运行 | 耗时 ms | 吞吐量 images/s |
| ---: | ---: | ---: |
| 1 | 50.691 | 631.28 |
| 2 | 49.347 | 648.46 |
| 3 | 52.428 | 610.36 |

---

## 36. 中位数汇总

| 模式 | Workers | 中位耗时 ms | 中位吞吐量 images/s | 相对顺序加速比 |
| --- | ---: | ---: | ---: | ---: |
| sequential | 1 | 123.279 | 259.57 | 1.000× |
| threads | 1 | 125.592 | 254.79 | 0.982× |
| threads | 2 | 79.611 | 401.96 | 1.549× |
| threads | 4 | 50.691 | 631.28 | 2.432× |

加速比：

```text
speedup = 顺序耗时 / 当前模式耗时
```

四线程：

```text
123.279 / 50.691 ≈ 2.432
```

因此在本次特定端到端基准中，四线程中位耗时约为顺序模式的41.1%，加速比约2.432倍。

该数字不能直接推广到其他图片、操作、磁盘、CPU或真实项目。

---

## 37. 如何解释本次结果

### 37.1 为什么1线程包装略慢

`threads / 1` 中位数125.592 ms，顺序模式123.279 ms。

加速比0.982，小于1，表示略慢。

原因包括：

- 创建工作线程；
- 原子索引读取；
- stop标记检查；
- join开销；
- 异常同步结构；
- 操作系统调度。

它说明并行框架本身不是免费的。

### 37.2 为什么2线程没有达到2倍

2线程加速比1.549，而不是2。

因为总耗时中包含：

- JPEG读写；
- 串行准备和汇总；
- 线程与原子开销；
- 缓存和内存带宽竞争；
- 不完全均匀的任务耗时。

### 37.3 为什么4线程没有达到4倍

4线程加速比2.432，而不是4。

并行加速通常受串行部分和共享资源限制。即使处理循环可以并行，程序仍然存在不能无限加速的部分。

这与 Amdahl 定律的直觉一致：只要程序有串行部分，整体加速就有上限。

### 37.4 为什么4线程仍明显快于2线程

本次32张图片的任务数量足以分给4个worker，且单张 Canny + JPEG I/O 的工作量能够覆盖部分线程管理成本。

但这只说明本次运行环境下的结果，不代表8或12线程会继续按相同比例提升。

---

## 38. 初测数据为什么没有作为最终结论

在固定 OpenCV 内部线程之前，曾得到初步中位数：

```text
sequential: 123.20 ms
threads-1: 126.58 ms
threads-2: 83.71 ms
threads-4: 62.39 ms
```

复核后发现，这组数据没有明确控制 OpenCV 内部并行，因此不能清晰区分：

- 外层 `std::thread` 的效果；
- OpenCV内部线程的效果；
- 两者嵌套后的效果。

于是新增测试并固定 `cv::setNumThreads(1)`，重新执行全部正确性和性能测量。

初测不是“无用失败”，它暴露了实验设计中的混杂变量。正确做法是保留问题和修正过程，而不是只留下最好看的数字。

---

## 39. 本节不能得出的结论

本节不能证明：

- 四线程对所有图片都能加速2.432倍；
- 线程越多越快；
- OpenCV内部并行一定比外层线程差；
- 当前实现是最优线程池；
- 磁盘不影响结果；
- 真实森林图像任务会得到相同比例；
- Debug构建的数字等于Release部署性能。

本节能证明的是：

- 当前线程分发设计在本次测试中保持输出一致；
- 1/2/4线程均能稳定完成32张图片；
- 在控制 OpenCV 内部线程后，本次四线程端到端中位耗时低于顺序模式；
- 测试和原始数据可以重新运行。

---

## 40. Debug 与 Release 性能

本节使用 Debug 构建，因为当前学习重点仍包括：

- 警告检查；
- 可调试性；
- 正确性验证。

Debug 性能数据适合比较同一构建中的不同模式，但不代表最终部署性能。

正式性能评估通常还应：

1. 增加 Release preset；
2. 使用相同输入；
3. 使用相同计时边界；
4. 重新进行预热和重复测量；
5. 不把 Debug 和 Release 数字混为同一实验。

Day51 暂不增加 Release 目标，避免同时引入过多变量。

---

## 41. 为什么没有实现完整线程池

当前程序每次调用 `process_directory()` 都创建一组线程，结束后 join。

真正长期运行的服务可能使用线程池：

- 线程提前创建；
- 任务进入队列；
- worker持续领取任务；
- 避免每批重复创建线程。

但线程池还涉及：

- 条件变量；
- 任务队列；
- 停止协议；
- 生命周期管理；
- 异常与取消；
- 背压。

Day51 的目标是理解最小正确线程模型，而不是一次完成通用线程池。

---

## 42. 为什么没有使用 `cv::parallel_for_`

OpenCV 提供 `cv::parallel_for_`，可以把一个 `cv::Range` 拆分给并行后端。

本节选择 `std::thread`，原因是当前路线重点补强 C++：

- 直接观察线程创建；
- 直接理解 join；
- 直接设计任务分配；
- 直接面对共享状态和异常传播。

掌握这些基础后，再使用 `parallel_for_` 时更容易判断：

- 循环是否适合拆分；
- Lambda 捕获是否安全；
- 输出区域是否互不重叠；
- 嵌套并行是否会过度订阅。

`parallel_for_` 是更高层工具，不会消除理解数据竞争的必要性。

---

## 43. 主程序为什么仍然精简

`main()` 负责：

- 检查参数结构；
- 解析操作、模式和线程数；
- 验证路径；
- 创建 `ImagePipeline`；
- 打印 `PipelineReport`；
- 映射退出码。

`main()` 不包含：

- `std::thread`；
- `std::atomic`；
- `std::mutex`；
- `cv::Canny`；
- `cv::imwrite`。

线程实现仍然位于 `day51_pipeline` 静态库中，保持 Day50 建立的模块边界。

---

## 44. 常见错误与排查

### 44.1 程序随机崩溃或结果数量变化

检查是否在线程中并发 `push_back()`，或是否修改了 `images`、`results` 的大小。

### 44.2 输出文件数量小于输入数量

检查：

- 原子下标终止条件；
- 输出文件名是否重复；
- 某个线程是否异常退出；
- `join()` 是否全部执行；
- 汇总中的读取和写入失败。

### 44.3 程序结束时调用 `terminate`

检查是否存在仍然 joinable 的线程对象，或线程函数中的异常是否未捕获。

### 44.4 多线程结果顺序随机

不要按完成顺序 `push_back()`。使用输入下标写回预先定长的结果向量。

### 44.5 多线程日志混乱

不要从worker直接打印每张图片。保存结构化结果，join后统一输出。

### 44.6 线程越多反而越慢

先确认：

- 输入数量是否足够；
- 是否包含磁盘I/O；
- OpenCV内部线程是否受控；
- 是否进行了预热；
- 是否重复测量；
- CPU是否被其他程序占用；
- 是否创建了远多于任务数量的线程。

### 44.7 CMake 找不到线程库

检查：

```cmake
find_package(Threads REQUIRED)
```

以及：

```cmake
Threads::Threads
```

然后重新配置 CMake，而不只是重新编译。

---

## 45. 七个核心问题

### 问题一：数据竞争是怎么产生的？

当多个线程访问同一个内存位置，至少一个线程写入，而且缺少正确同步时会产生数据竞争。并发 `vector::push_back()`、普通整数自增和多个线程写同一个结果元素都是典型风险。

### 问题二：为什么原子索引不会让两个线程领取同一张图片？

`fetch_add(1)` 把“读取旧值并递增”作为原子操作执行。每次调用返回的旧值在原子修改顺序中唯一，因此不同线程不会得到同一个成功领取的下标。

### 问题三：为什么不能让多个线程同时 `push_back()`？

`push_back()` 会修改 `vector` 的大小、内部指针并可能重新分配存储。多个线程同时修改这些共享内部状态会产生数据竞争。`reserve()` 也不会保护大小更新。

### 问题四：`join()` 解决了什么问题？

`join()` 建立明确的完成点：调用线程等待工作线程结束。它保证在线程使用的局部变量销毁前，所有线程已经停止，也保证汇总和计时发生在工作全部结束后。

### 问题五：为什么先验证哈希，再比较速度？

因为错误程序可能通过少做、漏做或覆盖输出而变快。只有确认文件数量、名称和字节内容一致，性能比较才有意义。

### 问题六：为什么不能只记录最快一次？

最快一次可能只是缓存、调度或CPU状态最有利的偶然值。预热、重复测量和中位数更接近典型表现，并保留原始数据供复核。

### 问题七：为什么 `steady_clock` 更适合计时？

它是单调时钟，不受系统墙上时间调整影响。性能测量需要稳定计算两个时刻之间的间隔，而不是获取日历时间。

---

## 46. 本节优点与限制

### 46.1 优点

- 顺序与线程模式共用同一处理函数；
- 原子索引实现动态负载分配；
- 结果向量不并行扩容；
- 输出顺序确定；
- worker异常可以回传；
- OpenCV内部线程被控制；
- 输出经过哈希一致性验证；
- 基准保留全部正式运行数据；
- 空目录和非法线程数有明确行为；
- 主程序没有泄漏线程实现细节。

### 46.2 限制

- 每次处理目录都重新创建线程；
- 没有通用线程池；
- `cv::setNumThreads(1)` 是进程级设置；
- 同名不同扩展名输入仍可能产生输出名冲突；
- 计时包含磁盘和JPEG编解码；
- 基准图片是同一图片的副本；
- 只测了Debug构建；
- 只测了1、2、4线程；
- 没有测量CPU利用率、能耗或内存峰值；
- 哈希一致不等于形式化证明无数据竞争。

---

## 47. 与机器视觉项目的联系

机器视觉系统经常需要并行处理：

- 多相机帧；
- 多张离线图片；
- 多个独立ROI；
- 预处理与推理流水线；
- 后处理和结果编码。

但需要先判断任务是否独立。

适合并行：

```text
图片A处理不依赖图片B
不同输出文件互不覆盖
每个结果写入独立位置
```

不应直接并行：

```text
后一帧依赖前一帧状态
多个任务修改同一模型状态
多个线程写同一输出缓冲区
硬件接口要求严格调用顺序
```

Day51 的设计思路可以迁移到离线数据预处理和多图推理，但真实项目还要结合模型线程安全、GPU上下文和硬件SDK约束。

---

## 48. 今日核心记忆点

1. 并行优化必须先保证正确性。
2. 数据竞争是未定义行为，不能靠一次成功运行排除。
3. 普通 `next_index++` 不是线程安全的任务分配。
4. `fetch_add()` 可以原子领取唯一任务下标。
5. 并行写 `vector` 前先固定大小，每个线程写不同元素。
6. `reserve()` 不能让并发 `push_back()` 安全。
7. Lambda 引用捕获要求被引用对象在线程结束前存活。
8. `join()` 是线程生命周期和结果可见性的关键完成点。
9. 不捕获线程异常会导致 `std::terminate()`。
10. `exception_ptr` 可以把worker异常带回调用线程。
11. 不在worker中直接打印，join后统一汇总。
12. `steady_clock` 用于测量时间间隔。
13. 一次测量不可靠，应预热、重复并记录中位数。
14. 先比较输出哈希，再比较速度。
15. 控制 OpenCV 内部线程，避免基准混杂。
16. 1线程包装可能比直接顺序循环更慢。
17. 4线程不会自动获得4倍加速。
18. 本次2线程加速1.549倍、4线程加速2.432倍，只适用于当前实验条件。
19. `hardware_concurrency()` 是参考值，不是最优线程数答案。
20. 生成数据、构建产物和基准CSV放在 `build/`，不上传GitHub。

---

## 49. Day52 预告

下一步计划：CMake 安装、打包与正式单元测试。

重点可能包括：

- `install(TARGETS ...)`；
- 安装公开头文件；
- 构建树与安装树的区别；
- `BUILD_INTERFACE` / `INSTALL_INTERFACE`；
- CTest；
- Catch2 或轻量测试目标；
- 对参数解析、操作名转换和线程数计算进行真正的C++单元测试；
- 保留现有PowerShell端到端契约测试。

Day52 会区分：

```text
单元测试：验证小函数和类行为
集成/契约测试：验证完整可执行程序、文件和退出码
```
