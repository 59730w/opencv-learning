# Day19：矩阵计算 \+ 自动求导

核心主题：梯度、反向传播、计算图、detach 分离、控制流求导

## 一、自动求导

自动求导（autograd）是 PyTorch 中**神经网络训练的基础设施**。一句话概括：

只需定义前向计算，PyTorch 自动追踪每一步运算，构建计算图，调用 `.backward()` 一键算出所有参数的梯度。

## 二、requires\_grad\_ — 开启梯度追踪

```python
import torch
x = torch.arange(4.0)         # tensor([0., 1., 2., 3.])
x.requires_grad_(True)         # 开启梯度追踪
print(x.grad)                  # None ← 此时还没计算梯度
```

|关键点|说明|
|---|---|
|requires\_grad\_(True\)|原地操作（带下划线 \_），标记该张量需要梯度|
|初始 x\.grad|在调用 \.backward\(\) 之前是 None|
|梯度存储位置|存储在 \.grad 属性中，不会自动清零|

## 三、标量反向传播

### 3\.1 示例：y = 2·(x·x\)

```python
x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)    # y = 2 × (0²+1²+2²+3²) = 2×14 = 28
y.backward()                # 反向传播
print(x.grad)               # tensor([ 0.,  4.,  8., 12.])
print(x.grad == 4 * x)      # tensor([True, True, True, True])
```

### 3\.2 手算验证

$y = 2(x_0^2 + x_1^2 + x_2^2 + x_3^2)$

对 $x_i$ 求偏导：

$\frac{\partial y}{\partial x_i} = 2 \cdot 2x_i = 4x_i$

|$x_i$|0|1|2|3|
|---|---|---|---|---|
|$\frac{\partial y}{\partial x_i} = 4x_i$|0|4|8|12|

✅ x\.grad 结果为 [0\., 4\., 8\., 12\.\]，完全吻合。

## 四、梯度清零 \.grad\.zero\_\(\)

PyTorch 的梯度默认累加，不清零会导致梯度叠加。每次反向传播前必须显式清零。

```python
x.grad.zero_()       # 梯度归零（原地操作）
y = x.sum()          # y = x₀ + x₁ + x₂ + x₃
y.backward()         # ∂y/∂xᵢ = 1
print(x.grad)        # tensor([1., 1., 1., 1.])
```

🔑 为什么不自动清零？ 某些场景（如 RNN 中跨时间步累积梯度）需要手动控制清零时机。

## 五、非标量输出：必须转为标量再反向传播

PyTorch 要求 \.backward\(\) 的调用对象必须是标量（0维张量）。

```python
x.grad.zero_()
y = x * x            # y shape: (4,)，非标量！
# y.backward()       # ❌ 直接调用会报错
y.sum().backward()   # ✅ 先 sum 变成标量，再反向传播
print(x.grad)        # tensor([0., 2., 4., 6.])
```

📐 验证：$y_i = x_i^2$，$\frac{\partial (\sum y_i)}{\partial x_i} = 2x_i$，即 [0, 2, 4, 6\] ✅

## 六、detach\(\) — 从计算图中分离

detach\(\) 返回一个共享数据但不参与梯度计算的新张量：切断反向传播路径。

```python
x.grad.zero_()
y = x * x              # y 在计算图中
u = y.detach()         # u 共享 y 的数据，但不在计算图中
z = u * x              # 反向传播时，梯度不会流经 u → y
z.sum().backward()
print(x.grad == u)     # tensor([True, True, True, True])
# 因为 ∂(u·x)/∂x = u（把 u 当作常数）
```

### 对比：不使用 detach

```python
x.grad.zero_()
y = x * x
y.sum().backward()     # 直接对 y.sum() 求导
print(x.grad == 2 * x) # tensor([True, True, True, True])
# ∂(x²)/∂x = 2x
```

### detach vs no detach 对照

|场景|计算图|梯度结果 x\.grad|解释|
|---|---|---|---|
|u = y\.detach\(\); z = u\*x; z\.sum\(\)\.backward\(\)|斩断|u = x²|u 被当作常数|
|y\.sum\(\)\.backward\(\)|完整|2x|y = x² 参与求导|

### detach\(\) 典型用途

|场景|说明|
|---|---|
|固定预训练模型|冻结 backbone，只训练分类头|
|GAN 训练|更新生成器时 detach 判别器输出|
|提取中间特征做可视化|只看特征不看梯度|
|避免梯度循环|某些复杂结构中手动控制梯度流向|

## 七、控制流也能求导

PyTorch 的 autograd 是动态计算图——Python 代码中的 if、while、for 都能正确求导。每次前向计算时记录实际执行路径，反向传播沿该路径回推。

### 7\.1 示例：含 while 循环的函数

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:   # 循环次数取决于 a 的值
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)  # 标量
d = f(a)
d.backward()
print(a.grad == d / a)  # True
```

### 7\.2 为什么 a\.grad == d/a？

分析计算过程：

每次循环 b = b \* 2，最终 b = a × 2^n（n 为总乘2次数）
无论分支如何走，d = k × a（k 为某常数）
因此 $\frac{\partial d}{\partial a} = k = \frac{d}{a}$

🧠 关键认识：控制流不是障碍。 autograd 记录执行路径，只要前向计算可微，反向传播自动处理。

## 八、课后练习详解

### 练习2：重复 backward 报错

```python
x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()              # ✅ 第一次成功
# y.backward()            # ❌ RuntimeError: 计算图已被释放
```

📌 原因：默认情况下 \.backward\(\) 执行后会释放中间缓冲区（retain\_graph=False）。如需多次反向传播，需设置 y.backward(retain_graph=True)。

### 练习3：向量输入的控制流求导

```python
a_vec = torch.randn(3, requires_grad=True)
d_vec = f(a_vec)          # 控制流函数，支持向量输入
d_vec.backward()          # 对向量每个元素独立求导
```

验证：a\_vec.grad == d\_vec / a\_vec，控制流对向量同样有效。

### 练习4：自定义控制流

```python
def custom_func(x):
    out = x
    for _ in range(3):
        if out.sum() < 5:    # 条件随 out 动态变化
            out = out + x
        else:
            out = out * x
    return out

t = torch.tensor([1.0, 2.0], requires_grad=True)
res = custom_func(t)
res.sum().backward()
t.grad  # 梯度自动算出，无需手动推导
```

### 练习5：绘制 sin(x\) 及其导函数

```python
import matplotlib.pyplot as plt
x_plot = torch.linspace(-torch.pi, torch.pi, 100, requires_grad=True)
f = torch.sin(x_plot)
f.sum().backward()         # 同时算所有点的导数
df = x_plot.grad           # cos(x)

plt.plot(x_plot.detach(), f.detach(), label='f(x)=sin(x)')
plt.plot(x_plot.detach(), df.detach(), label="f'(x)=cos(x)")  # autograd求出的导函数
plt.legend()
plt.grid(True)
```

🎯 直观验证：sin(x\) 的波峰处（±π/2）导数为 0，过零点处导数为 ±1，autograd 精确还原。

## 九、计算图生命周期

前向传播 → 构建计算图（记录每一步操作）
│
▼
\.backward\(\) → 反向传播（沿图反向计算梯度）
│
▼
释放计算图（默认行为，释放内存）

|参数|效果|
|---|---|
|默认（retain\_graph=False）|backward 后释放图，再次调用报错|
|retain\_graph=True|保留计算图，可多次 backward|
|create\_graph=True|构建高阶导数计算图（用于求二阶导）|

## 十、torch\.no\_grad\(\) 推理模式

评估/推理时禁用梯度追踪，节省显存、加速计算：

```python
with torch.no_grad():
    y = model(x)  # 不构建计算图，不追踪梯度
```

## 十一、本日核心速查表

|编号|操作|代码|说明|
|---|---|---|---|
|①|开启梯度|x\.requires\_grad\_(True\)|下划线表示原地操作|
|②|反向传播|y\.backward\(\)|y 必须是标量|
|③|查看梯度|x\.grad|初始为 None|
|④|梯度清零|x\.grad\.zero\_\(\)|默认累加，每次训练前必须清零|
|⑤|非标量反向|y\.sum\(\)\.backward\(\)|先转标量再反向传播|
|⑥|分离计算图|u = y\.detach\(\)|数据共享，梯度不回流|
|⑦|保留计算图|y\.backward(retain\_graph=True\)|可多次 backward|
|⑧|推理模式|with torch\.no\_grad\(\):|不追踪梯度|
|⑨|控制流求导|直接用 if/while|PyTorch 动态图自动处理|

## 十二、一张图梳理 autograd 流程

requires\_grad=True
│
▼
前向计算（构建动态计算图）
│
▼
\.backward\(\)（标量调用）
│
▼
\.grad 存储梯度（累加模式！）
│
▼
\.grad\.zero\_\(\)（手动清零）
│
▼
下一轮前向计算 \.\.\.

## 十三、关键记忆点

|序号|知识点|一句话|
|---|---|---|
|①|标量 backward|y 必须是标量，非标量先 \.sum\(\) 再 \.backward\(\)|
|②|梯度累加|不会自动清零，每次训练循环开始必须 \.grad\.zero\_\(\)|
|③|detach|斩断梯度流向，共享数据不共享计算图——冻结网络的关键操作|
|④|动态计算图|Python 控制流（if/while/for）天然支持求导|
|⑤|retain\_graph|保留计算图才能多次 backward，否则默认释放|
|⑥|梯度验证|x\.grad == 4\*x 这种手算验证明白每一步梯度的数学含义|
