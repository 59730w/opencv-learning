# Day28：使用 GPU 训练

**核心主题**：CUDA 检测 → 张量在 CPU/GPU 间移动 → `try_gpu()` 设备选择器 → 模型搬上 GPU → 同一设备运算规则

---

## 一、为什么需要 GPU？

|对比维度|CPU|GPU|
|---|---|---|
|核心数|4\~16 个大核|数千个小核|
|擅长|复杂逻辑、串行任务|大规模矩阵运算、并行任务|
|神经网络训练|慢（尤其大模型）|快 10\~100 倍|
|典型操作|数据处理、控制流|卷积、矩阵乘法、批量归一化|

深度学习训练的本质是大量的矩阵乘法和卷积运算——这正是 GPU 最擅长的事。一个 `nn.Linear(1000, 5000)` 的矩阵乘法，GPU 上可能只需要几毫秒，CPU 上耗时会高出几十倍。

---

## 二、检测 CUDA 环境

```python
import torch

print("CUDA 是否可用:", torch.cuda.is_available())

# True  → 已安装 GPU 版 PyTorch + NVIDIA 驱动正常
# False → CPU 版 PyTorch 或驱动不可用
```

### 2\.1 获取 GPU 详细信息

```python
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))   # 例如：NVIDIA GeForce RTX 3050
    print("GPU 数量:", torch.cuda.device_count())       # 例如：1
    print("当前设备:", torch.cuda.current_device())     # 例如：0
```

|函数|返回值类型|作用说明|
|---|---|---|
|`torch.cuda.is_available()`|bool|判断 GPU 是否可用|
|`torch.cuda.device_count()`|int|返回可用 GPU 数量|
|`torch.cuda.get_device_name(i)`|str|返回第 i 块 GPU 的型号名称|
|`torch.cuda.current_device()`|int|返回当前默认 GPU 的索引|

---

## 三、设备对象：torch\.device

```python
print(torch.device('cpu'))    # device(type='cpu')
print(torch.device('cuda'))   # device(type='cuda')    ← 等价于 cuda:0
print(torch.device('cuda:0')) # device(type='cuda', index=0)
print(torch.device('cuda:1')) # device(type='cuda', index=1) ← 第 2 块 GPU
```

`torch.device` 是 PyTorch 中的「设备地址标签」，用于精准指定张量、模型运行的硬件设备。

---

## 四、设备选择器：try\_gpu\(\) 和 try\_all\_gpus\(\)

两个核心工具函数，用于编写**设备无关代码**，一份代码兼容 CPU/GPU 环境。

```python
def try_gpu(i=0):
    """如果存在第 i 块 GPU，返回它；否则返回 CPU"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用 GPU，无 GPU 则返回 [cpu]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
```

|调用方式|有 1 块 GPU 时|无 GPU 时|
|---|---|---|
|`try_gpu()`|cuda:0|cpu|
|`try_gpu(10)`|cpu（无第11块GPU）|cpu|
|`try_all_gpus()`|[cuda:0\]|[cpu\]|

🔑 **核心优势**：替代硬编码 `cuda`，代码可在任意设备环境无缝运行，不会报错。

---

## 五、张量的设备属性

### 5\.1 默认设备

```python
x = torch.tensor([1, 2, 3])
print(x.device)  # device(type='cpu')  ← 默认创建在 CPU 上
```

所有无指定设备的新建张量，默认均创建在 CPU 上。

### 5\.2 创建时指定设备

```python
X = torch.ones(2, 3, device=try_gpu())   # 一步到位创建在 GPU 上
Y = torch.rand(2, 3, device=try_gpu(0))  # 指定 0 号 GPU
```

|参数写法|示例|效果|
|---|---|---|
|`device=try_gpu()`|`torch.zeros(..., device=try_gpu())`|有GPU则在GPU创建，无则CPU|
|`device='cuda'`|`torch.ones(..., device='cuda')`|强制在默认GPU创建|
|不写device参数|`torch.randn(...)`|默认在CPU创建|

---

## 六、张量在 CPU/GPU 间迁移

### 6\.1 \.cuda\(\) 和 \.cpu\(\)

```python
X = torch.ones(2, 3, device=try_gpu())   # X 在 GPU 上
Z = X.cuda(0)    # 迁移至 GPU:0（已在该设备则无操作）
X.cpu()          # 张量搬回 CPU
```

### 6\.2 设备判等核心实验

```python
Z = X.cuda(0)
print(Z.cuda(0) is Z)  # True
```

🧠 **核心逻辑**：若张量已处于目标设备，`.cuda(i)` 直接返回原对象，零开销；仅设备不一致时，才执行数据复制。

---

## 七、同一设备运算（核心强制规则）

### 7\.1 运算规则

**参与运算的所有张量必须位于同一设备，否则直接报错**，PyTorch 不会自动迁移数据。

```python
# 同设备运算（合法）
X = torch.ones(2, 3, device=try_gpu())
Y = torch.rand(2, 3, device=try_gpu(0))
print(X + Y)   # ✅ 正常运算

# 跨设备运算（非法）
a = torch.tensor([1, 2, 3])               # CPU 张量
b = torch.tensor([4, 5, 6], device='cuda') # GPU 张量
# a + b                                    # ❌ RuntimeError 报错
```

⚠️ 该机制为 PyTorch 安全设计，规避隐式数据传输带来的未知性能损耗。

### 7\.2 跨设备运算解决方案：显式迁移

```python
a = a.to(device='cuda')   # 手动迁移至GPU
a + b                      # ✅ 设备统一，正常运算
```

---

## 八、模型搬到 GPU

### 8\.1 模型设备迁移：net\.to(device\)

```python
import torch.nn as nn
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())    # 整体模型迁移至目标设备
```

该方法会递归遍历网络所有子模块，将全部 `nn.Parameter`（权重、偏置）迁移至目标设备。

### 8\.2 验证模型设备

```python
print(net[0].weight.data.device)
# device(type='cuda', index=0)  ← 参数已成功迁移至GPU
```

### 8\.3 GPU 训练完整流程

① 模型迁移GPU：`net = net.to(try_gpu())`

    ② 数据迁移GPU：`X, y = X.to(try_gpu()), y.to(try_gpu())`

    ③ 前向计算：`y_hat = net(X)`（GPU端执行）

    ④ 反向传播：`l.backward()`（GPU端计算梯度）

    ⑤ 参数更新：`optimizer.step()`（GPU端原地更新参数）

    ⑥ 结果读取：`loss.item()`（标量自动回传CPU）

---

## 九、GPU 训练通用模版（后续所有代码基础）

```python
# 1. 初始化设备
device = try_gpu()

# 2. 模型初始化并迁移设备
net = MyModel()
net.to(device)

# 3. 训练循环
for X, y in data_iter:
    # 数据迁移至GPU
    X, y = X.to(device), y.to(device)
    # 前向传播
    y_hat = net(X)
    l = loss(y_hat, y)
    # 反向传播与参数更新
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
```

---

## 十、往期课程的设备适配改造

Day16\-Day27 所有代码默认运行在CPU，从本课开始需适配GPU训练，改造规则如下：

|原有CPU写法|改造后GPU写法|
|---|---|
|`net = nn.Sequential(...)`|`net = nn.Sequential(...).to(try_gpu())`|
|`X = torch.randn(...)`|`X = torch.randn(..., device=try_gpu())` 或 `X.to(try_gpu())`|
|DataLoader直接取值 `X,y`|`X, y = X.to(device), y.to(device)`|

📌 强制习惯：后续所有训练代码，第一步优先定义`device = try_gpu()`，确保模型、数据设备统一。

---

## 十一、GPU 训练核心注意事项

|注意事项|详细说明|
|---|---|
|显存按需加载|每个batch数据用完即刻释放，禁止一次性将全部数据迁入GPU，避免显存溢出|
|item\(\) 自动回传CPU|`loss.item()` 可自动将GPU标量转为CPU浮点数，开销极低，无需手动迁移|
|numpy\(\) 强制先迁CPU|GPU张量无法直接转numpy，必须执行 `tensor.cpu().numpy()`|
|多GPU训练|复杂多卡场景使用 `DataParallel` / `DistributedDataParallel`，进阶再深入学习|
|显存不足解决方案|减小batch\_size、简化模型结构、降低网络宽度|

---

## 十二、与往期课程的关联适配

|往期课程|GPU适配改动点|
|---|---|
|Day20 线性回归|训练循环增加 `X.to(device)`、`y.to(device)`|
|Day22 MLP网络|模型初始化后增加 `net.to(try_gpu())`|
|Day26 Kaggle房价预测|所有 `.numpy()` 操作前需添加`.cpu()`|
|Day27 模型保存加载|`state_dict()` 仅保存参数数值，与设备无关，加载后可自由迁移CPU/GPU|

---

## 十三、本日关键记忆点

|编号|知识点|一句话总结|
|---|---|---|
|①|`torch.cuda.is_available()`|GPU环境检测第一步，判断CUDA是否可用|
|②|`try_gpu()`|通用设备选择器，自适应GPU/CPU，实现设备无关代码|
|③|张量默认设备|新建张量默认在CPU，需手动指定设备才可创建在GPU|
|④|同设备运算规则|CPU与GPU张量不可直接运算，必须统一设备|
|⑤|`.cuda(i)` 零开销机制|张量已在目标设备则返回自身，无复制开销|
|⑥|`net.to(device)`|递归迁移模型所有参数至目标设备|
|⑦|GPU张量转numpy|必须执行 `.cpu().numpy()`，两步缺一不可|
|⑧|`loss.item()`|GPU标量可自动回传CPU，无需手动迁移设备|
