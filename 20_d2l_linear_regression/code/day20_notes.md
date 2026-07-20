# Day20：线性回归——从零实现 \+ 简洁实现

核心主题：线性回归从零实现（数据生成→模型→损失→SGD→训练循环）→ 用 `nn.Linear` \+ `nn.MSELoss` \+ `torch.optim.SGD` 简洁重写

## 一、从零实现（scratch）

### 1\.1 生成模拟数据集

```python
def synthetic_data(w, b, num_examples):
    """
    生成 y = Xw + b + 噪声
    w: 真实权重向量
    b: 真实偏置
    num_examples: 样本数量
    """
    X = torch.normal(0, 1, (num_examples, len(w)))  # 特征：N(0,1) 正态分布
    y = torch.matmul(X, w) + b                        # 线性模型
    y += torch.normal(0, 0.01, y.shape)              # 添加微小噪声
    return X, y.reshape((-1, 1))
```

设置真实参数：

```python
true_w = torch.tensor([2, -3.4])  # 两个特征的真实权重
true_b = 4.2                       # 真实偏置
features, labels = synthetic_data(true_w, true_b, 1000)
# features.shape: (1000, 2)
# labels.shape:   (1000, 1)
```

数学模型：$\mathbf{y} = \mathbf{X}\mathbf{w} + b + \epsilon,\quad \epsilon \sim \mathcal{N}(0, 0.01^2)$

### 1\.2 可视化第二个特征与标签的关系

```python
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()
```

选择 features[:, 1\]（第二个特征）绘制散点图，w[1\] = \-3\.4，呈现明显的负相关趋势。

### 1\.3 小批量数据迭代器

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)          # 每个 epoch 随机打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]
```

|步骤|作用|
|---|---|
|random\.shuffle|打乱顺序，破坏相邻样本相关性|
|yield|生成器模式，每次返回一个小批量，节省内存|
|min(i\+batch\_size, num\_examples\)|处理最后一个不完整批次|

🔑 小批量随机梯度下降（Minibatch SGD）：每次用一个 batch（如10个样本）估计梯度，在计算效率和梯度稳定性之间取得平衡。

### 1\.4 初始化模型参数

```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # N(0, 0.01²)
b = torch.zeros(1, requires_grad=True)                        # 初始化为 0
```

|参数|形状|初始化方式|requires\_grad|
|---|---|---|---|
|w|(2, 1\)|正态分布 N(0, 0\.01²\)|✅ True|
|b|(1,\)|零|✅ True|

小方差初始化让初始预测接近0，配合线性模型从零开始收敛。

### 1\.5 线性回归模型

```python
def linreg(X, w, b):
    return torch.matmul(X, w) + b
```

数学形式：$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$

$\mathbf{X} \in \mathbb{R}^{batch\_size \times 2}$：小批量特征矩阵
$\mathbf{w} \in \mathbb{R}^{2 \times 1}$：权重列向量
$b \in \mathbb{R}$：标量偏置（广播到每一行）
$\hat{\mathbf{y}} \in \mathbb{R}^{batch\_size \times 1}$：预测值

### 1\.6 均方损失函数（MSE）

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

数学形式：$\ell(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$

|注意点|说明|
|---|---|
|除以2|求导后常数消掉，梯度变成 $(\hat{y}-y)$，形式上更干净|
|y\.reshape(y\_hat\.shape\)|确保 y 和 y\_hat 形状一致，防止广播异常|
|返回值|返回的是每个样本的损失向量（非标量），形状 (batch\_size, 1\)|

### 1\.7 小批量随机梯度下降

```python
def sgd(params, lr, batch_size):
    with torch.no_grad():          # 参数更新不追踪梯度
        for param in params:
            param -= lr * param.grad / batch_size  # 沿负梯度方向更新
            param.grad.zero_()                      # 清空梯度，防止累加
```

更新公式：$\theta \leftarrow \theta - \frac{\eta}{batch\_size} \cdot \nabla_{\theta} L$

|组件|含义|
|---|---|
|lr|学习率 η，控制步长|
|param\.grad|当前参数的梯度（已在 backward\(\) 中计算好）|
|/ batch\_size|除以 batch\_size 取平均——因为损失在小批量上做了 sum 再 backward|
|with torch\.no\_grad\(\)|参数更新是原地操作，不构建计算图|
|param\.grad\.zero\_\(\)|梯度清零，为下一轮累积做准备|

⚠️ 为什么除以 batch\_size？l\.sum\(\)\.backward\(\) 对整个 batch 的损失和求导，求出来的梯度是 batch 内所有样本梯度之和。除以 batch\_size 取平均，让学习率与 batch 大小解耦。

### 1\.8 完整的训练循环（核心流程）

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)   # ① 前向计算
        l.sum().backward()          # ② 反向传播求梯度
        sgd([w, b], lr, batch_size) # ③ 参数更新
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

### 1\.9 训练循环步骤拆解

每个 epoch：
┌─────────────────────────────────┐
│ for 每个 batch:                   │
│   ① 前向：l = loss(net(X,w,b\),y\) │
│   ② 反向：l\.sum\(\)\.backward(\)   │
│   ③ 更新：sgd([w,b\], lr, bs\)      │
│       ├ param \-= lr\*grad/bs       │
│       └ grad\.zero\_\(\)           │
│                                   │
│ 计算全量训练集损失（评估用）       │
└─────────────────────────────────┘

### 1\.10 训练结果与参数恢复

```python
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
# 例如：tensor([ 0.0004, -0.0008])  ← 非常接近 true_w = [2, -3.4]
print(f'b的估计误差: {true_b - b}')
# 例如：tensor([0.0005])            ← 非常接近 true_b = 4.2
```

仅 3 个 epoch 就几乎完全恢复了真实参数，线性回归的凸优化收敛非常快。

## 二、简洁实现（concise）——使用 torch\.nn 模块

### 2\.1 数据加载：DataLoader \+ TensorDataset

```python
from torch.utils import data

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)  # 把特征和标签打包成数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

|从零实现|简洁实现|优势|
|---|---|---|
|手写 data\_iter \+ random\.shuffle \+ yield|DataLoader \+ TensorDataset|多线程加载、自动批处理、pin\_memory 加速|

### 2\.2 搭建网络：nn\.Linear(2, 1\)

```python
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
# 输入维度=2，输出维度=1 —— 就是线性层 y = Xw + b

net[0].weight.data.normal_(0, 0.01)  # 权重 N(0, 0.01²)
net[0].bias.data.fill_(0)            # 偏置 0
```

|从零实现|简洁实现|
|---|---|
|手动定义 w, b 张量 \+ requires\_grad=True|nn\.Linear(2, 1\) 自动管理参数和梯度|
|手写 linreg(X, w, b\)|net(X\) 直接调用|

### 2\.3 损失函数：nn\.MSELoss\(\)

```python
loss = nn.MSELoss()  # 均方误差，等价于 squared_loss 但不除以 2
```

|从零实现|简洁实现|
|---|---|
|squared\_loss：除以2，求和后手动 backward|nn\.MSELoss：不除以2，自动兼容 backward|

### 2\.4 优化器：torch\.optim\.SGD

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

|从零实现|简洁实现|
|---|---|
|手写 sgd(params, lr, batch\_size\)|trainer\.zero\_grad\(\) \+ trainer\.step\(\)|
|手动 param\.grad\.zero\_\(\) \+ 手动更新|优化器内部管理梯度清零和参数更新|

### 2\.5 简洁版训练循环

```python
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)        # ① 前向
        trainer.zero_grad()        # ② 梯度清零
        l.backward()               # ③ 反向传播
        trainer.step()             # ④ 参数更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

## 三、从零 vs 简洁 四步对照

|步骤|从零实现|简洁实现|
|---|---|---|
|数据|手写 data\_iter 生成器|DataLoader \+ TensorDataset|
|模型|linreg(X, w, b\) 手动矩阵乘|nn\.Sequential(nn\.Linear(2,1\)\)|
|损失|squared\_loss 手写/2|nn\.MSELoss\(\)|
|优化|sgd\(\) 手动梯度下降|torch\.optim\.SGD|

🧠 学习建议：从零实现是为了理解原理（每个环节发生了什么），简洁实现是日常写法（以后写模型都用这个）。

## 四、线性回归数学模型汇总

|组件|数学形式|代码|
|---|---|---|
|模型|$\hat{y} = \mathbf{X}\mathbf{w} + b$|torch\.matmul(X, w\) \+ b|
|损失|$\ell = \frac{1}{2}(\hat{y} - y)^2$|(y\_hat \- y\) \*\* 2 / 2|
|梯度|$\nabla_w \ell = \mathbf{X}^\top (\hat{y} - y)$|autograd 自动计算|
|更新|$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{B} \nabla_w L$|param \-= lr \* param\.grad / batch\_size|

$B$ = batch_size，$L$ = batch 内损失之和

## 五、训练流程统一模版

```python
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)      # ① 前向：算损失
        optimizer.zero_grad()    # ② 清零：清空旧梯度
        l.backward()             # ③ 反向：算梯度
        optimizer.step()         # ④ 更新：梯度下降一步
    # 每个 epoch 结束时评估
```

🎯 这四行代码是 PyTorch 训练循环的核心范式，从线性回归到 Transformer，结构完全一样。

## 六、关键记忆点

|编号|知识点|一句话|
|---|---|---|
|①|训练流程四步|前向→清零→反向→更新，所有模型通用|
|②|为什么 /batch\_size|sum\(\)\.backward\(\) 得出的是 batch 内梯度和，除以 batch\_size 取平均|
|③|with torch\.no\_grad\(\)|参数更新不追踪梯度，手写 SGD 必须包裹|
|④|梯度清零位置|在 backward\(\) 之前、上一轮 step\(\) 之后——简洁版在 step\(\) 前调 zero\_grad\(\)|
|⑤|y\.reshape(y\_hat\.shape\)|防止标签广播异常的关键防御性编程|
|⑥|3 epoch 收敛|线性回归是凸优化问题，小学习率 \+ 几步 SGD 即可逼近全局最优|
