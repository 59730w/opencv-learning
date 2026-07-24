# Day24：权重衰减 \+ 暂退法/Dropout

**核心主题**：L2正则化（权重衰减）→ Dropout（暂退法）→ 两种防过拟合技术的原理、从零实现与简洁使用

**过拟合工具箱正式建立！**

---

# 第一部分：权重衰减（Weight Decay）

## 1\.1 核心思想

过拟合的根源：模型过于复杂，参数值过大，对训练集中每个细微噪声都敏感。

**权重衰减的解决思路**：在损失函数中额外加上一项惩罚——参数越大，惩罚越重。迫使模型用尽量小的参数去拟合数据。

## 1\.2 数学形式

原始损失（MSE）：

$Loss_{original} = \frac{1}{2} \cdot (\hat{y} - y)^2$

加上 L2 惩罚后的损失：

$Loss = Loss_{original} + \frac{\lambda}{2} \cdot \|w\|^2$

其中：

- **λ**（lambda）是超参数，控制惩罚力度

- $\|w\|^2 = w_1^2 + w_2^2 + ... + w_n^2$，即所有权重的平方和

- 除以 2 是为了求导后常数消掉，梯度变成 $梯度原项 + \lambda \cdot w$，形式上更干净

## 1\.3 为什么 L2 正则化叫"权重衰减"？

SGD 更新公式（加了 L2 惩罚后）：

$w \leftarrow w - \eta \cdot (\frac{\partial Loss_{original}}{\partial w} + \lambda \cdot w)$

拆开来看：

$w \leftarrow (1 - \eta \cdot \lambda) \cdot w - \eta \cdot \frac{\partial Loss_{original}}{\partial w}$

括号内为**衰减因子**

每次更新，先让 w 缩小一点（乘以 $1-\eta\lambda$），再沿原始梯度方向更新。

🔑 这就是为什么叫 **Weight Decay**——权重在每次更新时自动"衰减"。

## 1\.4 实验设计：高维小样本的线性回归

真实数据有 200 个特征，但只有 20 个训练样本——**典型的过拟合高发场景**（特征数远多于样本数）。

```python
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((200, 1)) * 0.01, 0.05
# 真实权重全为 0.01，偏置为 0.05
```

|条件|数值|含义|
|---|---|---|
|训练样本|20|很少|
|测试样本|100|足够评估泛化能力|
|特征数|200|远多于训练样本 → 极易过拟合|
|真实 w|全 0\.01|一个小而均匀的权重分布|

## 1\.5 从零实现关键代码

L2 惩罚项：

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2   # Σ wᵢ² / 2
```

带惩罚的损失计算（核心一行）：

```python
l = loss(net(X), y) + lambd * l2_penalty(w)
#   \_____________/     \___________________/
#      原始MSE损失          L2惩罚项 × 超参数λ
```

完整对比：

|组件|无正则化（Day20）|有权重衰减（Day24）|
|---|---|---|
|损失|loss(net(X\), y\)|loss(net(X\), y\) \+ λ · l2\_penalty(w\)|
|梯度|仅来自 MSE|来自 MSE \+ λ·w|
|效果|参数可能很大|参数被压制在小值|

## 1\.6 三组对比实验

|实验|λ (lambd\)|w 的 L2 范数|现象|
|---|---|---|---|
|无正则化|0|很大（≈13）|过拟合，test loss 分叉上扬|
|中等正则化|3|较小（≈0\.4）|泛化较好|
|强正则化|5|非常小|更平滑，但如果 λ 太大可能欠拟合|

λ = 0 时 w 的 L2 范数很大——模型"自由发挥"，把训练集的噪声也拟合进去了。

λ = 3/5 时 w 的 L2 范数明显缩小——模型被强制用小声量去描述数据，被迫只学真正的规律。

📐 L2 范数：`torch.norm(w)` 计算$\sqrt{\sum w_i^2}$，越大说明权重越"发散"。

## 1\.7 简洁实现：weight\_decay 参数

在 PyTorch 优化器中直接指定 weight\_decay，无需手动修改损失函数：

```python
trainer = torch.optim.SGD([
    {"params": net[0].weight, 'weight_decay': wd},   # 权重：施加衰减
    {"params": net[0].bias}                           # 偏置：不施加衰减
], lr=lr)
```

关键点说明

- 只对 weight 加 weight\_decay

- bias 通常不需要正则化

- wd 就是 λ，和从零实现的 lambd 对应

- 内部自动处理，每次 trainer\.step\(\) 自动加上 $\lambda \cdot w$ 的梯度

⚠️ bias 不加权重衰减是标准做法——偏置只是一个平移量，不会导致过拟合。

## 1\.8 从零 vs 简洁对照

|从零实现|简洁实现|
|---|---|
|L2 惩罚手动 `lambd * l2_penalty(w)` 加进 loss|`optim.SGD(..., weight_decay=wd)`|
|参数分组手动传 `[w, b]`|`{"params": weight, "weight_decay": wd} + {"params": bias}`|
|`l.sum().backward() + sgd(...)`|`trainer.zero_grad() + l.mean().backward() + trainer.step()`|

# 第二部分：暂退法（Dropout）

## 2\.1 核心思想

一个团队里如果有几个"明星员工"包揽一切，其他人就得不到锻炼；明星一走，团队就瘫痪了。Dropout 的做法是每次训练随机让一部分神经元"休息"，逼着每个神经元都要学会独当一面。

换个说法：Dropout 在训练时随机丢弃神经元，相当于每次都在训练一个不同的子网络。测试时所有神经元都参与，等同于对大量子网络做集成平均。

## 2\.2 Dropout 的数学操作

设丢弃概率为 p（即参数 dropout），则：

**训练时**：以概率 p 将神经元输出置 0，然后把保留的值放大 $\frac{1}{1-p}$ 倍

**测试时**：什么也不做，所有神经元正常工作

为什么要放大 $\frac{1}{1-p}$？

因为训练时只有 $(1-p)$ 比例的神经元在工作，输出期望值缩小了。放大 $\frac{1}{1-p}$ 可以让训练和测试时的输出期望保持一致。

举例（p = 0\.5，即丢弃一半）：

训练时的某个神经元：

- 50% 概率：输出变成 0（被丢弃）

- 50% 概率：输出变成 原始值 × 2（保留并放大）

- 期望值 = $0.5 \times 0 + 0.5 \times (原始值 \times 2) = 原始值$ ← 和测试时一致！

## 2\.3 手写 Dropout 实现

```python
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)     # 全丢弃 → 全零
    if dropout == 0:
        return X                        # 不丢弃 → 原样
    mask = (torch.rand(X.shape) > dropout).float()   # 随机掩码
    return mask * X / (1.0 - dropout)                # 丢弃 + 缩放
```

逐行解析：

- `torch.rand(X.shape)` → 生成 \[0,1\) 均匀分布随机数，形状和 X 一样

- `> dropout` → 大于 dropout 的置 True（保留），小于的置 False（丢弃）

- `.float()` → True/False 转 1\.0/0\.0

- `mask * X` → 被丢弃的神经元输出清零

- `/ (1.0 - dropout)` → 保留下来的值放大，保持期望一致

## 2\.4 Dropout 测试样例

```python
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
# X = [[0, 1, 2, 3, 4, 5, 6, 7],
#      [8, 9,10,11,12,13,14,15]]

print(dropout_layer(X, 0.))   # 全保留，原样输出
print(dropout_layer(X, 0.5))  # 约一半置 0，保留的翻倍
print(dropout_layer(X, 1.))   # 全零
```

🔑 只在训练时用 Dropout，测试/推理时不用！ `net.eval()` 会自动关闭 `nn.Dropout`。

## 2\.5 从零实现带 Dropout 的网络

```python
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)   # 784 → 256
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)  # 256 → 256
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)   # 256 → 10
        self.relu = nn.ReLU()
        self.training = is_training    # 控制训练/推理模式

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, dropout1)   # 第一个隐藏层后 dropout=0.2
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)   # 第二个隐藏层后 dropout=0.5
        return self.lin3(H2)
```

## 2\.6 Dropout 概率的常见设置

|层位置|常用 dropout 值|原因|
|---|---|---|
|输入层附近|0\.1 \~ 0\.2|靠近数据，不宜丢弃太多信息|
|中间隐藏层|0\.5|对中间特征做较强的正则化|
|输出层附近|0\.0 \~ 0\.2|靠近分类决策，保留足够信息|

实验中：

```python
dropout1, dropout2 = 0.2, 0.5
# 第一个隐藏层后丢弃 20%
# 第二个隐藏层后丢弃 50%
```

## 2\.7 简洁实现：nn\.Dropout

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),        # ← 一行，自动处理训练/测试切换
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.5),        # ← 一行
    nn.Linear(256, 10)
)
```

nn\.Dropout 自动行为：

|模式|行为|触发方式|
|---|---|---|
|train|随机丢弃神经元|训练循环前调用 `net.train()`|
|eval|什么都不做，直接透传|评估/推理时调用 `net.eval()`|

在之前 `train_epoch_ch3` 和 `evaluate_accuracy` 中，已经写好了 `net.train()` 和 `net.eval()` 的切换，所以 nn\.Dropout 会自动配合，无需手动干预。

# 第三部分：权重衰减 vs Dropout 对比

|维度|权重衰减 (L2\)|Dropout|
|---|---|---|
|作用对象|损失函数（加惩罚项）|神经元输出（随机置零）|
|机制|压制单个参数的绝对值|打破神经元间的共适应|
|直观理解|"参数别太大"|"别让少数神经元说了算"|
|训练时|损失多了 $\lambda \cdot \|w\|^2$ 项|随机丢神经元|
|测试时|损失只用原始项|关闭 dropout，全部神经元参与|
|适用范围|所有参数化模型|主要用在全连接层|
|PyTorch 简洁方式|weight\_decay 参数|nn\.Dropout(p\) 层|
|超参数|λ（权重衰减系数）|p（丢弃概率）|

# 第四部分：防过拟合工具箱（完整版）

Day23 学习了诊断，Day24 学习了治疗——现在有了完整的武器库：

|方法|适用场景|Day|
|---|---|---|
|增加训练数据|数据量不足导致过拟合|Day23 提及|
|降低模型容量|模型远复杂于任务需要|Day23|
|L2 / 权重衰减|参数值过大|Day24|
|Dropout|神经元共适应|Day24|
|早停 (Early Stopping\)|验证 loss 不再下降时停|后续|
|数据增广|图像/文本等可变换数据|后续|

# 第五部分：关键代码速查

## 5\.1 权重衰减 — 从零实现（一行改损失）

```python
l = loss(net(X), y) + lambd * (torch.sum(w.pow(2)) / 2)
```

## 5\.2 权重衰减 — 简洁实现（一行改优化器）

```python
trainer = torch.optim.SGD([
    {"params": weight, 'weight_decay': wd},
    {"params": bias}                    # bias 不加
], lr=lr)
```

## 5\.3 Dropout — 从零实现（核心逻辑）

```python
mask = (torch.rand(X.shape) > dropout).float()
return mask * X / (1.0 - dropout)
```

## 5\.4 Dropout — 简洁实现（一行加层）

```python
nn.Dropout(0.5)
```

# 第六部分：本日关键记忆点

|编号|知识点|一句话|
|---|---|---|
|①|权重衰减原理|损失函数加 $\lambda \cdot \|w\|^2$，迫使权重变小|
|②|为什么叫"衰减"|更新公式中 $w \leftarrow (1-\eta\lambda)\cdot w - \eta\cdot梯度$，自动缩小|
|③|bias 不加|偏置平移不导致过拟合，通常不施加 weight\_decay|
|④|Dropout 机制|训练时随机丢神经元 → 打破共适应 → 测试时全用|
|⑤|为什么放大 1/(1\-p\)|保持训练和测试时输出期望一致|
|⑥|train/eval 切换|nn\.Dropout 自动响应：训练时丢，测试时透传|
|⑦|常见 dropout 值|输入层 0\.1\~0\.2，中间层 0\.5，输出层 0\~0\.2|
|⑧|L2 vs Dropout|L2 压制权重大小；Dropout 打破神经元依赖——两者互补|