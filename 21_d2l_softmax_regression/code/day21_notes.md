# Day21：图像分类数据集 \+ Softmax回归

核心主题：Fashion\-MNIST 数据集 → Softmax 函数 → 交叉熵损失 → 多分类训练全流程

## 第一部分：图像分类数据集（Fashion\-MNIST）

### 1\.1 数据集简介

|属性|值|
|---|---|
|名称|Fashion\-MNIST|
|图像尺寸|28 × 28 灰度图|
|训练集|60,000 张|
|测试集|10,000 张|
|类别数|10|

### 1\.2 10 个类别标签

```python
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
```

|索引|0|1|2|3|4|5|6|7|8|9|
|---|---|---|---|---|---|---|---|---|---|---|
|标签|T恤|裤子|套头衫|连衣裙|外套|凉鞋|衬衫|运动鞋|包|短靴|

### 1\.3 加载数据集

```python
import torchvision
from torchvision import transforms

trans = transforms.ToTensor()  # PIL Image → Tensor，像素值 0~1
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True,  transform=trans, download=True)
mnist_test  = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

|参数|说明|
|---|---|
|root|数据存放路径（\.\./data 放在项目外部，避免污染 repo）|
|train=True/False|训练集 / 测试集|
|transform=trans|数据预处理管线|
|download=True|首次运行自动下载，已有则不重复下载|

### 1\.4 transforms\.ToTensor\(\) 做了什么

|转换|说明|
|---|---|
|PIL Image (H × W × C，uint8，0\~255\)|→|
|Tensor (C × H × W，float32，0\.0\~1\.0\)|通道前置 \+ 归一化|

### 1\.5 单张图片的形状

```python
mnist_train[0][0].shape  # torch.Size([1, 28, 28])
# C × H × W = 1 × 28 × 28
```

### 1\.6 可视化批量图片

```python
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())  # Tensor → NumPy 才能绘图
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # 隐藏坐标轴
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])  # 显示标签文字
    return axes
```

使用示例：

```python
X, y = next(iter(DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
```

关键细节：X 形状是 (18, 1, 28, 28\)，imshow 需要 (H, W\) 或 (H, W, C\)，所以 reshape(18, 28, 28\)。

### 1\.7 DataLoader 多进程加速

```python
train_iter = DataLoader(mnist_train, batch_size=256, shuffle=True,
                         num_workers=4)
```

|参数|作用|
|---|---|
|batch\_size=256|每次取 256 张|
|shuffle=True|每个 epoch 随机打乱|
|num\_workers=4|4 个子进程并行加载，显著加快读取速度|

### 1\.8 封装完整加载函数（可复用模板）

```python
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))  # Resize 放在 ToTensor 之前
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True,  transform=trans, download=True)
    mnist_test  = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True,  num_workers=4),
            DataLoader(mnist_test,  batch_size, shuffle=False, num_workers=4))
```

注意：Compose 中操作的顺序很重要——

Resize(resize\) — 先改变尺寸（操作 PIL Image）
ToTensor\(\) — 再转张量

## 第二部分：Softmax 回归从零实现（scratch）

理论定位：Softmax 回归 = 线性回归 \+ Softmax 函数 \+ 交叉熵损失，把输出从「单个数值」变成「10 个类别的概率分布」。

### 2\.1 模型参数初始化

```python
num_inputs  = 784   # 28 × 28 展平
num_outputs = 10    # 10 个类别
W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
```

### 2\.2 Softmax 函数

```python
def softmax(X):
    X_exp = torch.exp(X)              # ① 指数化：所有值变正数
    partition = X_exp.sum(1, keepdim=True)  # ② 每行求和（归一化分母）
    return X_exp / partition          # ③ 归一化：保证每行和为 1
```

数学形式（对第 i 个样本的第 j 类）：

softmax(x_j) = exp(x_j) / sum_k exp(x_k)

|步骤|代码|目的|
|---|---|---|
|exp|torch\.exp(X\)|把任意实数映射到正数区间|
|sum(1, keepdim=True\)|沿列方向求和|得到每行的归一化分母|
|exp / partition|广播除法|每行变成概率分布，和为 1|

验证：

```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob.sum(1))   # tensor([1.0000, 1.0000])
```

### 2\.3 网络模型定义

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

流程：(batch, 1, 28, 28\) → reshape(\-1, 784\) → 矩阵乘 → \+b → softmax → \(batch, 10\)

### 2\.4 交叉熵损失函数

交叉熵衡量「预测分布」与「真实分布」之间的距离。

```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
```

取预测概率的技巧：

```python
y = torch.tensor([0, 2])                     # 真实标签
y_hat = torch.tensor([[0.1, 0.3, 0.6],       # 样本0：类别0概率=0.1
                      [0.3, 0.2, 0.5]])      # 样本1：类别2概率=0.5
y_hat[[0, 1], y]  # [y_hat[0,0], y_hat[1,2]] = [0.1, 0.5]
```

数学形式（对第 i 个样本，真实类别为 $y_i$）：

Loss = -log( p_yi )，其中 p_yi 是模型对真实类别的预测概率

|预测概率 p|\-log(p\)|含义|
|---|---|---|
|0\.9|0\.105|✅ 预测准确，损失很小|
|0\.5|0\.693|不太确定|
|0\.1|2\.303|❌ 几乎预测错了，损失很大|

### 2\.5 准确率计算

```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 取概率最大的类别
    cmp = y_hat.type(y.dtype) == y     # 逐元素比较
    return float(cmp.type(y.dtype).sum())
```

### 2\.6 模型评估函数

```python
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()                       # 切换到评估模式
    metric = Accumulator(2)              # [正确数, 总数]
    with torch.no_grad():                # 不计算梯度
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]         # 正确数 / 总数
```

### 2\.7 单轮训练函数

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()                      # 切换到训练模式
    metric = Accumulator(3)              # [损失和, 正确数, 样本数]
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()          # nn 损失：取均值再反向
            updater.step()
        else:
            l.sum().backward()           # 手写损失：求和再反向
            updater(X.shape[0])          # 手写更新器传入 batch_size
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
    # 返回：(平均损失, 训练准确率)
```

### 2\.8 完整训练函数（含动态绘图）

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss< 0.5       # 损失应低于 0.5
    assert train_acc > 0.7        # 训练准确率 > 70%
    assert test_acc  > 0.7        # 测试准确率 > 70%
```

### 2\.9 训练执行

```python
lr = 0.1
num_epochs = 10
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

### 2\.10 预测可视化

```python
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break                          # 取第一个 batch
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[:n].reshape(n, 28, 28), 1, n, titles=titles)
```

每张图片上方显示「真实标签 \\n 预测标签」，错误预测一目了然。

## 第三部分：Softmax 回归简洁实现（concise）

### 3\.1 网络搭建

```python
from torch import nn

net = nn.Sequential(
    nn.Flatten(),        # (batch, 1, 28, 28) → (batch, 784)
    nn.Linear(784, 10)   # (batch, 784) → (batch, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # N(0, 0.01²)
        m.bias.data.fill_(0)                 # 偏置 = 0
net.apply(init_weights)  # 递归应用到所有子模块
```

|层|作用|输入形状|输出形状|
|---|---|---|---|
|nn\.Flatten\(\)|展平|(batch, 1, 28, 28\)|(batch, 784\)|
|nn\.Linear(784,10\)|全连接 \+ softmax（在 Loss 内部）|(batch, 784\)|(batch, 10\)|

⚠️ 关键差异：简洁版网络中没有手动写 softmax。nn\.CrossEntropyLoss\(\) 内部已包含 LogSoftmax \+ NLLLoss，所以网络的最后一层只需输出原始的 logits（未归一化得分）。

### 3\.2 损失函数

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

|参数|含义|
|---|---|
|reduction='none'|返回每个样本的损失向量（非平均），与从零实现保持一致|
|reduction='mean'（默认）|自动对 batch 取均值|

内部等价流程：CrossEntropyLoss = LogSoftmax \+ NLLLoss

### 3\.3 为什么 nn\.CrossEntropyLoss 不需要手动 softmax？

从零实现：  Linear → Softmax → log → NLL → Loss

简洁实现：  Linear → CrossEntropyLoss（内置 LogSoftmax \+ NLLLoss）

Softmax 和 log 合并为 LogSoftmax，数值稳定性更好（避免 exp 溢出）

### 3\.4 优化器与训练

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(10):
    for X, y in train_iter:
        l = loss(net(X), y)       # CrossEntropyLoss 自动处理 softmax
        trainer.zero_grad()
        l.mean().backward()       # reduction='none' 时手动 mean
        trainer.step()
```

## 第四部分：从零 vs 简洁：Softmax 回归对照

|步骤|从零实现|简洁实现|
|---|---|---|
|数据|d2l\.load\_data\_fashion\_mnist|同上|
|模型|softmax(matmul(reshape\(X\), W\) \+ b\)|nn\.Sequential(nn\.Flatten\(\), nn\.Linear(784,10\)\)|
|Softmax|手写 softmax(X\) 函数|内置于 nn\.CrossEntropyLoss\(\)|
|损失|手写 cross\_entropy(y\_hat, y\)|nn\.CrossEntropyLoss(reduction='none'\)|
|优化|手写 sgd \+ with torch\.no\_grad\(\)|torch\.optim\.SGD|
|准确率|手写 accuracy \+ evaluate\_accuracy|同上逻辑|

## 第五部分：关键工具类

### 5\.1 Accumulator（累加器）

```python
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
```

|用途|示例|
|---|---|
|训练评估|metric = Accumulator(3\) 分别跟踪 [损失和, 正确数, 样本数\]|
|最终输出|metric[0\] / metric[2\] = 平均损失|

### 5\.2 Animator（PyCharm 适配动态绘图）

```python
animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                    ylim=[0.3, 0.9],
                    legend=['train loss', 'train acc', 'test acc'])
animator.add(epoch, (train_loss, train_acc, test_acc))
```

替代 d2l\.Animator 的 PyCharm 兼容版本，关键改动：

使用 plt\.draw\(\) \+ plt\.pause(0\.01\) 替代 Jupyter 的 display\.clear\_output
手动管理 self\.X / self\.Y 列表存储历史数据

## 第六部分：线性回归 vs Softmax 回归 对比

|线性回归（Day20）|Softmax 回归（Day21）|
|---|---|
|任务：回归（预测数值）|任务：分类（预测类别）|
|输出层：1 个神经元|输出层：10 个神经元|
|输出含义：连续实数值|输出含义：每个类别的概率分布|
|激活函数：无（恒等）|激活函数：Softmax（归一化为概率）|
|损失函数：MSE（均方误差）|损失函数：交叉熵（Cross\-Entropy）|
|标签形状：(batch, 1\)|标签形状：(batch,\) — 类别索引|
|matplotlib 图：散点图（特征 vs 标签）|matplotlib 图：训练曲线（loss \+ acc）|

## 第七部分：交叉熵 vs MSE 为什么分类用交叉熵？

|损失函数|分类问题表现|原因|
|---|---|---|
|MSE|梯度消失（饱和区梯度过小）|Softmax 输出接近 0/1 时，导数趋近 0|
|交叉熵|梯度稳定，收敛快|log 抵消了 softmax 的 exp，梯度 = p \- y（线性）|

🧠 直觉：交叉熵 \+ Softmax 的组合在数学上是「最自然」的——梯度等于「预测概率 \- 真实标签」，形状简单、信号强。

## 第八部分：训练流程统一视角

线性回归（Day20）：  Linear(2,1\) → MSE Loss ← 数值标签，连续值输出

Softmax回归（Day21）：Linear(784,10\) → CrossEntropy ← 类别索引，logits 输出（Loss 内置 softmax）

但训练循环四步完全一样：

```python
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)        # ① 前向
        optimizer.zero_grad()      # ② 清零
        l.backward()               # ③ 反向（l 需是标量或 .mean()/.sum() 后）
        optimizer.step()           # ④ 更新
```

## 第九部分：本日关键记忆点

|编号|知识点|一句话|
|---|---|---|
|①|Fashion\-MNIST|灰度 28×28，10 类，是图像分类的「Hello World」|
|②|ToTensor\(\)|PIL → Tensor \+ 归一化到 [0,1\] \+ 通道前置|
|③|DataLoader|batch\_size \+ shuffle \+ num\_workers 三项组合|
|④|Softmax|exp 归一化，把 logits 变成概率分布，每行和为 1|
|⑤|交叉熵|\-log(p\_true\)，与 Softmax 配合梯度 = p \- y|
|⑥|CrossEntropyLoss|内置 LogSoftmax \+ NLLLoss，网络端不要加 softmax|
|⑦|reduction='none'|返回逐样本损失向量，手动 \.mean\(\) 或 \.sum\(\)|
|⑧|训练循环不变|线性回归 / Softmax 回归 / 后续 CNN → 四步完全一样|
