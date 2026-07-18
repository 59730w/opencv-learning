# Day18：线性代数

核心主题：标量/向量/矩阵/张量 → 求和与均值 → 点积/矩阵乘法 → 范数

## 一、标量运算

标量（Scalar）是单个数值，0维张量。

```python
import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)
x + y   # 5.0
x * y   # 6.0
x / y   # 1.5
x ** y  # 9.0  ← 幂运算
```

## 二、向量基础

向量（Vector）是1维张量，有方向、有长度。

|属性|代码|结果|
|---|---|---|
|创建|x = torch\.arange(4\)|[0, 1, 2, 3\]|
|访问元素|x[3\]|3|
|长度|len(x\)|4|
|形状|x\.shape|torch\.Size([4\]\)|

## 三、矩阵与转置

矩阵（Matrix）是2维张量：(行数, 列数\)。

### 3\.1 创建矩阵

```python
A = torch.arange(20).reshape(5, 4)  # 5行4列，0~19按行填入
```

### 3\.2 转置 \.T

```python
A.T  # shape: (4, 5)，行列互换
```

### 3\.3 对称矩阵检验

```python
B = torch.tensor([[1, 2, 3],
                  [2, 0, 4],
                  [3, 4, 5]])
B == B.T  # 全 True → B 是对称矩阵
```

🔑 对称矩阵：$A\_\{ij\} = A\_\{ji\}$，A == A\.T 全为 True。

## 四、三维张量

三维张量常用于：批量图像数据 (批量大小, 高, 宽\)。

```python
X = torch.arange(24).reshape(2, 3, 4)
# 2个3×4矩阵，shape: (2, 3, 4)
```

## 五、矩阵按元素运算

⚠️ 关键区分：\* 是逐元素乘法，不是矩阵乘法！

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 深拷贝，互不影响
A + B  # 逐元素加法
A * B  # 逐元素乘法（Hadamard积），不是 A @ B！
```

## 六、标量与张量的广播

标量会自动广播到张量的每一个元素。

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X          # 每个元素 +2
(a * X).shape  # torch.Size([2, 3, 4])，形状不变
```

## 七、求和操作

### 7\.1 基本求和

```python
x = torch.arange(4, dtype=torch.float32)
x.sum()  # 0+1+2+3 = 6.0
A.sum()  # 所有元素求和 = 0+1+...+19 = 190.0
```

### 7\.2 按轴求和

|操作|结果 shape|直观理解|
|---|---|---|
|A\.sum(axis=0\)|(4,\)|沿行方向压缩：5行→消去行，保留4列|
|A\.sum(axis=1\)|(5,\)|沿列方向压缩：4列→消去列，保留5行|
|A\.sum(axis=[0,1\]\)|标量|等价于 A\.sum\(\)，全部消去|

```python
# A.shape = (5, 4)
A.sum(axis=0)       # shape: (4,)   ← 每一列的和（跨行求和）
A.sum(axis=1)       # shape: (5,)   ← 每一行的和（跨列求和）
A.sum(axis=[0, 1])  # 标量 190.0   ← 全体求和
```

🧠 记忆口诀：axis=n 就是消去第 n 个维度。axis=0 消去行维，axis=1 消去列维。

## 八、均值

```python
A.mean()                  # 全体均值
A.sum() / A.numel()       # 等价写法，numel = 元素总数
A.mean(axis=0)            # 每列的均值，shape: (4,)
A.sum(axis=0) / A.shape[0]  # 等价写法
```

### 求和 vs 均值速查

|操作|语义|
|---|---|
|A\.sum(\)|所有元素和|
|A\.mean(\)|所有元素均值 = A\.sum\(\) / A\.numel\(\)|
|A\.sum(axis=0\)|沿 axis=0 压缩求和|
|A\.mean(axis=0\)|沿 axis=0 压缩求均值|

## 九、保持维度求和 keepdims=True

这是广播除法的关键技巧。

```python
sum_A = A.sum(axis=1, keepdims=True)
# A.shape:      (5, 4)
# sum_A.shape:  (5, 1) ← 保持维度，不是 (5,)！
A / sum_A  # 每行除以该行的和，shape: (5, 4)
```

### 对比：

|写法|结果 shape|能否直接参与广播除法|
|---|---|---|
|A\.sum(axis=1\)|(5,\)|❌ 维度对不上|
|A\.sum(axis=1, keepdims=True\)|(5, 1\)|✅ 广播兼容|

🔑 keepdims=True 不消去被求和维度，保留为大小1的维度，方便后续广播。

## 十、累积求和 cumsum

```python
A.cumsum(axis=0)  # 沿行方向逐行累加
```

第0行：保持不变
第1行：第0行 \+ 第1行
第2行：第0行 \+ 第1行 \+ 第2行
……

常用于累积分布场景。

## 十一、点积

两个相同长度向量的点积 = 逐元素乘再求和。

```python
x = torch.arange(4, dtype=torch.float32)  # [0, 1, 2, 3]
y = torch.ones(4, dtype=torch.float32)    # [1, 1, 1, 1]
torch.dot(x, y)    # 0×1 + 1×1 + 2×1 + 3×1 = 6.0
torch.sum(x * y)   # 等价写法 —— 逐元素乘法再求和
```

公式：$\mathbf{x} \cdot \mathbf{y} = \sum_{i} x_i y_i$

## 十二、矩阵\-向量乘法

torch\.mv(A, x\)：矩阵 × 向量。

```python
A.shape  # (5, 4)
x.shape  # (4,)    ← 向量的长度必须等于矩阵的列数！
torch.mv(A, x)     # shape: (5,)  ← 输出长度等于矩阵的行数
```

公式：$\mathbf{c}_i = \sum_j A_{ij} \cdot x_j$

## 十三、矩阵\-矩阵乘法

torch\.mm(A, B\)：这才是真正的矩阵乘法，不是逐元素乘！

```python
A.shape  # (5, 4)
B = torch.ones(4, 3)
B.shape  # (4, 3)
torch.mm(A, B)  # shape: (5, 3)
# 内维一致方可乘：(5,4) × (4,3) → (5,3)
```

📐 形状法则：(m, n\) @ (n, p\) → (m, p\)，中间维度必须一致。

## 十四、三种"乘"的对照

|运算符/函数|语义|示例|
|---|---|---|
|A \* B|逐元素乘（Hadamard积）|(5,4\) \* (5,4\) → (5,4\)|
|torch\.dot(x, y\)|向量点积（仅1维）|(4,\) · (4,\) → 标量|
|torch\.mv(A, x\)|矩阵\-向量乘|(5,4\) × (4,\) → (5,\)|
|torch\.mm(A, B\)|矩阵\-矩阵乘（真正的矩阵乘法）|(5,4\) × (4,3\) → (5,3\)|

## 十五、范数

范数衡量向量/矩阵的"大小"。

### 15\.1 L2 范数（欧几里得范数）

torch\.norm(u\) 默认计算 L2 范数：

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)   # √(3² + (-4)²) = 5.0
```

公式：$\|\mathbf{u}\|_2 = \sqrt{\sum_i u_i^2}$

### 15\.2 L1 范数

```python
torch.abs(u).sum()  # |3| + |-4| = 7.0
```

公式：$\|\mathbf{u}\|_1 = \sum_i |u_i|$

### 15\.3 矩阵 Frobenius 范数

torch\.norm\(矩阵\) 计算 Frobenius 范数：

```python
torch.norm(torch.ones((4, 9)))
# √(4×9×1²) = √36 = 6.0
```

公式：$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$，本质是把矩阵拉成向量算 L2。

## 十六、L1 vs L2 快速对照

|维度|L1 范数|L2 范数|
|---|---|---|
|公式|$\sum \|x_i\|$|$\sqrt{\sum x_i^2}$|
|PyTorch|torch\.abs(u\)\.sum\(\)|torch\.norm(u\)|
|特点|产生稀疏解|产生平滑解|
|别名|曼哈顿距离|欧几里得距离|

## 十七、核心速查表

|编号|操作|PyTorch|注意|
|---|---|---|---|
|①|转置|A\.T|行列互换|
|②|逐元素乘|A \* B|⚠️ 不是矩阵乘法|
|③|沿轴求和|A\.sum(axis=0/1\)|axis=n 消去第 n 维|
|④|保持维度|keepdims=True|保留大小1维度，用于广播|
|⑤|点积|torch\.dot(x, y\)|仅限1维向量|
|⑥|矩阵\-向量乘|torch\.mv(A, x\)|(m,n\) × (n,\) → (m,\)|
|⑦|矩阵乘法|torch\.mm(A, B\)|(m,n\) × (n,p\) → (m,p\)|
|⑧|L2 范数|torch\.norm(u\)|默认欧几里得距离|
|⑨|L1 范数|torch\.abs(u\)\.sum(\)|绝对值求和|

## 十八、关键区分

|操作|说明|
|---|---|
|逐元素乘  A \* B|形状相同，对应位置相乘|
|矩阵乘法  torch\.mm|内维一致，(m,n\)@(n,p\)→(m,p\)|
|点积      torch\.dot|两个一维向量 → 标量|
|axis=0|消去第0维 → 列向量求和，保留列|
|axis=1|消去第1维 → 行向量求和，保留行|
|keepdims|保留维度 → shape从 (5,\) 变为 (5,1\)|
