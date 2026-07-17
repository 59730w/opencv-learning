# Day17：数据操作 \+ 数据预处理

核心主题：张量（Tensor）基本操作、广播机制、内存模型、pandas数据预处理

## 一、张量创建

张量（Tensor）是 PyTorch 中最基本的数据结构，可以理解为「多维数组」。

### 1\.1 `torch.arange()` — 生成等差数列

```python
import torch
x = torch.arange(12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

### 1\.2 张量的基本属性

|属性|含义|示例|
|---|---|---|
|\.shape|形状（各维度大小）|torch\.Size([12\]\)|
|\.numel\(\)|元素总数|12|
|\.reshape(m, n\)|改变形状（不改变数据）|(3, 4\)|

```python
print(x.shape)    # torch.Size([12])
print(x.numel())  # 12
X = x.reshape(3, 4)  # 3行4列矩阵
```

### 1\.3 特殊张量创建

|函数|用途|说明|
|---|---|---|
|torch\.zeros\((a, b, c\)\)|全0张量|常用于初始化偏置|
|torch\.ones\((a, b, c\)\)|全1张量|常用于掩码/缩放因子|
|torch\.randn(m, n\)|标准正态分布随机张量|常用于权重初始化|
|torch\.tensor(list\)|从 Python 列表手动创建|数据类型自动推断|

```python
zero  = torch.zeros((2, 3, 4))   # shape: 2×3×4
ones  = torch.ones((2, 3, 4))    # shape: 2×3×4
rand  = torch.randn(3, 4)        # shape: 3×4, N(0,1)
manul = torch.tensor([[2,1,4,3], [1,2,3,4], [4,3,2,1]])
```

## 二、张量运算

### 2\.1 逐元素四则运算

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y   # tensor([ 3.,  4.,  6., 10.])
x - y   # tensor([-1.,  0.,  2.,  6.])
x * y   # tensor([ 2.,  4.,  8., 16.])    ← 逐元素乘，不是矩阵乘法
x / y   # tensor([0.5, 1.0, 2.0, 4.0])
x ** y  # tensor([ 1.,  4., 16., 64.])    ← 逐元素幂运算
```

### 2\.2 指数运算

```python
torch.exp(x)  # e^x，逐元素计算
# tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
```

## 三、张量拼接

`torch.cat((A, B), dim=n)`：沿第 n 个维度拼接。

|dim 方向|直观理解|效果|
|---|---|---|
|dim=0|按行方向堆叠|行数增加，列数不变|
|dim=1|按列方向堆叠|列数增加，行数不变|

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.,1,4,3],[1,2,3,4],[4,3,2,1]])
cat_dim0 = torch.cat((X, Y), dim=0)  # shape: (6, 4)
cat_dim1 = torch.cat((X, Y), dim=1)  # shape: (3, 8)
```

### 3\.1 条件比较（逐元素，返回布尔张量）

```python
X == Y
# tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]])
```

### 3\.2 求和

```python
X.sum()  # 所有元素求和 → tensor(66.)
```

## 四、广播机制

广播（Broadcasting）允许不同形状的张量进行运算，规则是：从最后一个维度向前对齐，缺失维度或大小为1的维度会自动扩展。

### 经典示例

```python
a = torch.arange(3).reshape((3, 1))   # shape: (3, 1)
b = torch.arange(2).reshape((1, 2))   # shape: (1, 2)
c = a + b                              # shape: (3, 2)
```

**广播过程：**

a (3,1\) → 复制扩展为 (3,2\)
b (1,2\) → 复制扩展为 (3,2\)
然后逐元素相加

### 三维广播

```python
m = torch.randn(2, 1, 4)   # shape: (2, 1, 4)
n = torch.randn(1, 3, 4)   # shape: (1, 3, 4)
(m + n).shape               # torch.Size([2, 3, 4])
```

## 五、索引与切片

语法与 NumPy 完全一致。

|操作|含义|代码|
|---|---|---|
|X[\-1\]|最后一行|等价于 X[2, :\]|
|X[1:3\]|第1\~2行（左闭右开）|X[1:3, :\]|
|X[1, 2\] = 9|单元素赋值|修改第1行第2列|
|X[0:2, :\] = 12|区域赋值|前两行所有列全部改为12|

## 六、原地操作与内存模型

### 6\.1 赋值运算 ≠ 原地

```python
before = id(Y)      # 记录旧地址
Y = Y + X           # 创建新张量，Y指向新地址
id(Y) == before     # False ← 地址变了！
```

### 6\.2 真正原地：切片赋值

```python
Z = torch.zeros_like(Y)
old_id = id(Z)
Z[:] = X + Y        # 切片赋值，不创建新内存
id(Z) == old_id     # True ← 地址不变
```

### 6\.3 原地自增运算符

```python
X += Y              # 等价于 X[:] = X + Y
id(X) 不变          # True，真正的原地操作
```

⚠️ 关键结论：\+=、\*= 等运算符会原地修改；Y = Y \+ X 则会分配新内存。

## 七、Tensor ↔ NumPy 互转

```python
A = X.numpy()              # Tensor → NumPy
B = torch.tensor(A)        # NumPy → Tensor
type(A)  # <class 'numpy.ndarray'>
type(B)  # <class 'torch.Tensor'>
```

⚠️ 两者共享底层内存（CPU 张量时），修改一个会影响另一个。

## 八、标量提取

```python
a = torch.tensor([3.5])
a.item()    # 3.5 ← 提取为 Python float
float(a)    # 3.5
int(a)      # 3   ← 截断
```

## 九、数据预处理

使用 pandas 进行经典的数据预处理流水线。

### 9\.1 读取 CSV 并查看原始数据

```python
import pandas as pd
data = pd.read_csv('../data/house_tiny.csv')
```

原始数据（含缺失值）：

|NumRooms|Alley|Price|
|---|---|---|
|NA|Pave|127500|
|2|NA|106000|
|4|NA|178100|
|NA|NA|140000|

### 9\.2 分离特征与标签

```python
inputs  = data.iloc[:, 0:2]   # 前两列：NumRooms, Alley
outputs = data.iloc[:, 2]     # 第三列：Price
```

### 9\.3 数值缺失值处理：均值填充

```python
inputs = inputs.fillna(inputs.mean(numeric_only=True))
# NA → 均值 (2+4)/2 = 3
```

### 9\.4 类别特征处理：独热编码（One\-Hot）

```python
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
```

dummy\_na=True：把缺失值也作为一个类别
结果：Alley\_Pave、Alley\_nan 两列 0/1 指示

### 9\.5 转换为 PyTorch 张量

```python
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
```

### 9\.6 完整预处理流水线总结

CSV读取 → 特征/标签分离 → 数值列均值填充 → 类别列独热编码 → 转Tensor

## 十、本日关键记忆点

|编号|知识点|总结|
|---|---|---|
|①|张量属性|\.shape 看形状、\.numel\(\) 看总数、\.reshape\(\) 改形状|
|②|广播规则|从最后一维向前对齐，1 扩展为匹配大小|
|③|原地操作|Y = Y \+ X 新地址；X \+= Y 原地；Z[:\] = \.\.\. 原地|
|④|拼接方向|dim=0 行堆叠，dim=1 列堆叠|
|⑤|预处理流水线|fillna(均值\) → get\_dummies → tensor|
|⑥|内存共享|CPU 上的 Tensor 和 NumPy 数组共享底层内存|
