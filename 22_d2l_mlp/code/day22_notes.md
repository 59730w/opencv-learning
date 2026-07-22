# Day 22：多层感知机 MLP——从零实现与简洁实现

> 核心主题：感知机局限 → 多层感知机（MLP）→ 隐藏层 → ReLU 激活函数 → 从零手写两层网络 → 使用 `nn.Sequential` 简洁搭建

---

## 一、感知机的局限

### 1.1 感知机模型

感知机（Perceptron）是最早的人工神经元模型，输出只有 0 或 1。

### 1.2 致命缺陷：XOR 问题

感知机**无法解决 XOR（异或）问题**，因为它是线性分类器，而 XOR 数据点不能用一条直线分开。

| 问题 | AND | OR | XOR |
|---|---|---|---|
| 线性可分？ | ✅ 是 | ✅ 是 | ❌ 不是 |

> 🔑 **核心结论**：单层感知机只能解决线性可分问题。要突破这个限制，必须引入**隐藏层**和**非线性激活函数**。

---

## 二、多层感知机 MLP

### 2.1 结构全景图

MLP 在输入层和输出层之间插入**隐藏层（Hidden Layer）**：

```text
输入层（784 个神经元） → 隐藏层（256 个神经元） → 输出层（10 个神经元）

X                     H = ReLU(X·W₁ + b₁)        O = H·W₂ + b₂
(batch, 784)          (batch, 256)               (batch, 10)
```

### 2.2 数学公式

**第一步：输入 → 隐藏层**

$$
H = \operatorname{ReLU}(XW_1 + b_1)
$$

逐项含义：

| 符号 | 含义 | 具体形状或说明 |
|---|---|---|
| $X$ | 输入的展平图像（784 个像素值） | 256 张图 × 784 像素，即 `(256, 784)` |
| $W_1$ | 第一层权重矩阵 | `(784, 256)`；每个输入像素连接到 256 个隐藏神经元 |
| $b_1$ | 第一层偏置 | `(256,)`；广播加到每一行 |
| $XW_1+b_1$ | 线性变换结果 | `(256, 256)` |
| $\operatorname{ReLU}(\cdot)$ | 将负数变为 0，正数保持不变 | 引入非线性 |
| $H$ | 隐藏层输出 | `(256, 256)` |

**第二步：隐藏层 → 输出层**

$$
O = HW_2 + b_2
$$

| 符号 | 含义 | 具体形状或说明 |
|---|---|---|
| $H$ | 隐藏层激活输出 | `(256, 256)` |
| $W_2$ | 第二层权重矩阵 | `(256, 10)`；每个隐藏神经元连接到 10 个输出类别 |
| $b_2$ | 第二层偏置 | `(10,)`；广播加到每一行 |
| $O$ | 最终输出（logits，未经 softmax） | `(256, 10)` |

> 📐 随后将 $O$ 传入 `CrossEntropyLoss`，它会在内部完成 LogSoftmax 和交叉熵计算。

### 2.3 为什么必须使用激活函数？

如果没有激活函数，即将 ReLU 换成恒等映射：

$$
\begin{aligned}
H &= XW_1+b_1,\\
O &= HW_2+b_2\\
  &= (XW_1+b_1)W_2+b_2\\
  &= X(W_1W_2)+(b_1W_2+b_2)\\
  &= XW_{\text{new}}+b_{\text{new}}.
\end{aligned}
$$

这仍然等价于单层线性模型。

> 🎯 **结论**：无论堆叠多少层，如果没有非线性激活函数，多层网络都等价于单层线性模型。激活函数是神经网络“深度”的来源，每一层都可以学习输入的不同非线性变换。

---

## 三、ReLU 激活函数

### 3.1 定义

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

即：**负数全部变为 0，正数原样输出**。

### 3.2 Python 实现

```python
def relu(X):
    a = torch.zeros_like(X)   # 全零张量，形状和 X 相同
    return torch.max(X, a)    # 逐元素计算 max(X, 0)
```

### 3.3 ReLU 计算示例

```text
输入 X：     [-2.0, -1.0, 0.0, 1.0, 2.0]
输出 ReLU： [ 0.0,  0.0, 0.0, 1.0, 2.0]
               ↑     ↑    ↑    ↑    ↑
             变 0  变 0  边界  保留  保留
```

### 3.4 为什么使用 ReLU？

| 对比项 | Sigmoid / Tanh | ReLU |
|---|---|---|
| 梯度饱和 | ❌ 当 $x$ 远离 0 时，梯度趋近于 0 | ✅ 当 $x>0$ 时，梯度恒为 1 |
| 计算量 | ❌ 需要 `exp` 运算 | ✅ 只需一个 `max` 操作 |
| 稀疏性 | ❌ 输出密集 | ✅ 负数输出 0，天然稀疏 |
| 收敛速度 | 较慢 | 较快 |

> 🧠 **梯度消失（Vanishing Gradient）**：Sigmoid 在两端的导数近似为 0。多层反向传播时，梯度连乘后会趋近于 0，使浅层参数几乎无法更新。ReLU 在正半轴的梯度始终为 1，可在很大程度上缓解这个问题。

---

## 四、从零实现

### 4.1 参数初始化

```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(784, 256, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(256, requires_grad=True))
W2 = nn.Parameter(torch.randn(256, 10, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(10, requires_grad=True))

params = [W1, b1, W2, b2]
```

| 参数 | 形状 | 初始化 | 说明 |
|---|---|---|---|
| `W1` | `(784, 256)` | `randn × 0.01` | 小方差正态分布 |
| `b1` | `(256,)` | `zeros` | 偏置通常初始化为 0 |
| `W2` | `(256, 10)` | `randn × 0.01` | 同上 |
| `b2` | `(10,)` | `zeros` | 同上 |

> ⚠️ 这里使用 `nn.Parameter` 而不是普通张量，以明确将其注册为可训练参数，供优化器追踪和更新。

### 4.2 网络定义（核心）

```python
def net(X):
    X = X.reshape((-1, num_inputs))   # 展平：(256, 1, 28, 28) → (256, 784)
    H = relu(X @ W1 + b1)             # 隐藏层 + ReLU 激活
    return H @ W2 + b2                # 输出 logits，不执行 softmax
```

与 Day 21 的 Softmax 回归对比：

```text
Day 21 Softmax 回归：X → reshape → Linear → 10 个 logits
Day 22 MLP：        X → reshape → Linear → ReLU → Linear → 10 个 logits
```

> 📌 **关键差异**：MLP 增加了隐藏层和 ReLU。输出层不执行 softmax，而是将 logits 交给 `CrossEntropyLoss` 处理。

### 4.3 训练

```python
loss = nn.CrossEntropyLoss(reduction='none')
updater = torch.optim.SGD(params, lr=0.1)

train_ch3(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs=10,
    updater=updater,
)
```

训练循环的四个步骤与 Day 20、Day 21 完全相同：

```python
for X, y in train_iter:
    l = loss(net(X), y)        # ① 前向传播
    updater.zero_grad()        # ② 梯度清零
    l.mean().backward()        # ③ 反向传播
    updater.step()             # ④ 更新参数
```

---

## 五、简洁实现（Concise Implementation）

### 5.1 使用 `nn.Sequential` 搭建 MLP

```python
net = nn.Sequential(
    nn.Flatten(),          # (batch, 1, 28, 28) → (batch, 784)
    nn.Linear(784, 256),   # 全连接：784 → 256
    nn.ReLU(),             # 激活函数
    nn.Linear(256, 10),    # 全连接：256 → 10
)
```

### 5.2 逐层解读

| 网络层 | 作用 |
|---|---|
| `nn.Flatten()` | 将 $28\times28$ 的图片展平为 784 维向量 |
| `nn.Linear(784, 256)` | 将 784 个输入映射到 256 个隐藏神经元，参数为 `W(784×256) + b(256)` |
| `nn.ReLU()` | 逐元素计算 $\max(0,x)$，引入非线性 |
| `nn.Linear(256, 10)` | 将 256 个隐藏层输出映射到 10 个类别 logits，参数为 `W(256×10) + b(10)` |

### 5.3 权重初始化

```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 标准差为 0.01
        m.bias.data.fill_(0)                 # 偏置初始化为 0

net.apply(init_weights)  # 递归应用到每个子模块
```

`net.apply(fn)` 会递归遍历 `nn.Sequential` 的所有子层。如果子层是 `nn.Linear`，上述函数就会初始化它的权重和偏置。

### 5.4 训练

```python
batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

train_ch3(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs,
    trainer,
)
```

---

## 六、从零实现与简洁实现对照

| 步骤 | 从零实现 | 简洁实现 |
|---|---|---|
| 参数定义 | 手动定义 `W1`、`b1`、`W2`、`b2`，并包装为 `nn.Parameter` | 由 `nn.Linear` 自动管理 |
| 隐藏层 | `relu(X @ W1 + b1)` | `nn.Linear(784, 256)` + `nn.ReLU()` |
| 输出层 | `H @ W2 + b2` | `nn.Linear(256, 10)` |
| 权重初始化 | `torch.randn(...) * 0.01` | 通过 `net.apply` 调用 `nn.init.normal_` |
| 优化器 | `SGD(params, lr)` | `SGD(net.parameters(), lr)` |

---

## 七、模型进化三部曲（Day 20 → Day 21 → Day 22）

| 对比项 | 线性回归（Day 20） | Softmax 回归（Day 21） | MLP（Day 22） |
|---|---|---|---|
| 层数 | 1 层 | 1 层 | 2 层 |
| 隐藏层 | ❌ 无 | ❌ 无 | ✅ 有（256 个神经元） |
| 激活函数 | ❌ 无（恒等映射） | ❌ 无（恒等映射） | ✅ ReLU |
| 非线性 | ❌ 纯线性 | ❌ 纯线性 | ✅ 非线性变换 |
| 表达能力 | 弱 | 弱 | 强 |
| 简洁版核心代码 | `Flatten → Linear(784, 1)` | `Flatten → Linear(784, 10)` | `Flatten → Linear(784, 256) → ReLU → Linear(256, 10)` |
| 损失函数 | MSE | CrossEntropyLoss | CrossEntropyLoss |

```text
线性回归：    Flatten → Linear(784, 1)                         → MSELoss
Softmax 回归：Flatten → Linear(784, 10)                        → CrossEntropyLoss
MLP：         Flatten → Linear(784, 256) → ReLU → Linear(256, 10)
                                                              → CrossEntropyLoss
```

> 🎯 从代码量看，MLP 只多了一个 `nn.ReLU()` 和一个 `nn.Linear` 层，就从线性模型变成了真正的神经网络。但概念上的跨越是巨大的：模型从只能处理线性可分问题，迈入了通用函数逼近的领域。

---

## 八、关键记忆点

| 编号 | 知识点 | 一句话总结 |
|---|---|---|
| ① | XOR 问题 | 感知机无法解决线性不可分问题，因而需要隐藏层 |
| ② | 隐藏层 | 位于输入层和输出层之间，用于提取高阶特征 |
| ③ | 激活函数的必要性 | 没有激活函数，多层网络仍等价于单层线性模型 |
| ④ | ReLU | $\max(0,x)$；可缓解梯度消失、计算速度快，是常用激活函数 |
| ⑤ | `nn.Parameter` | 将张量标记为可训练参数，使优化器可以追踪和更新 |
| ⑥ | `net.apply(fn)` | 递归初始化所有子层，替代手动逐层赋值 |
| ⑦ | 输出层不放 softmax | `CrossEntropyLoss` 内置 LogSoftmax，网络只需输出 logits |
| ⑧ | 训练循环不变 | 前向传播 → 梯度清零 → 反向传播 → 参数更新，适用于从线性回归到 MLP 的训练 |
