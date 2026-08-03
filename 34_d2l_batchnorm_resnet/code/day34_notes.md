# Day34：批量归一化 \+ ResNet——训练深层网络的利器

**核心主题**：BatchNorm（批量归一化）从零实现 → 带 BN 的 LeNet → ResNet 残差块 → 1×1 卷积做维度匹配 → ResNet\-18 完整结构

---

## 第一部分：批量归一化 BatchNorm

### 1\.1 为什么需要 BatchNorm？

深层网络训练困难的核心原因：

|问题|说明|
|---|---|
|**内部协变量偏移**|前面层的参数更新后，后面层的输入分布会不断变化——每层都得重新适应|
|**梯度不稳定**|Day25 讲的梯度消失/爆炸——深层时连乘效应放大|
|**对初始化敏感**|初始参数稍微不好，训练就发散|
|**学习率不能太大**|大了梯度爆炸，小了收敛太慢|

BatchNorm 在每一层做一次"强制标准化"，让每层的输入都稳定在均值 0、方差 1 附近。

### 1\.2 BatchNorm 的数学流程

对一个 batch 的输入 X：

① 求 batch 均值：$\mu = mean(X, dim=0)$

② 求 batch 方差：$\sigma^2 = mean((X - \mu)^2, dim=0)$

③ 标准化：$\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2 + \varepsilon}} \quad (\varepsilon=1e-5 \text{ 防除零})$

④ 缩放和平移：$Y = \gamma \cdot \hat{X} + \beta$ （$\gamma$ 和 $\beta$ 是可学习参数）

**γ（gamma）和 β（beta）是关键**——如果标准化后直接输出，网络会失去表达能力（被强制约束在 N\(0,1\) 空间）。γ（初始=1）和 β（初始=0）让网络可以**学回**原始分布，甚至学出更好的分布。

### 1\.3 从零实现——全连接层 vs 卷积层的处理差异

```python
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        # ===== 推理模式：用全局统计量（moving_mean/var）=====
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # ===== 训练模式：用当前 batch 的统计量 =====
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层：X shape = (batch, features)
            mean = X.mean(dim=0)                        # 对 batch 维度求均值
            var  = ((X - mean) ** 2).mean(dim=0)        # 对 batch 维度求方差
        else:
            # 卷积层：X shape = (batch, channels, H, W)
            mean = X.mean(dim=(0, 2, 3), keepdim=True)  # 对 batch+H+W 求均值
            var  = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新全局统计量（指数移动平均）
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var  = momentum * moving_var  + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

```

|差异点|全连接层 \(2D\)|卷积层 \(4D\)|
|---|---|---|
|输入形状|(batch, features\)|(batch, channels, H, W\)|
|均值/方差的维度|mean(dim=0\) —— 对 batch 求|mean(dim=(0,2,3\)\) —— 对 batch\+H\+W 求|
|结果形状|(features,\)|(1, channels, 1, 1\) —— 每个通道一个均值/方差|
|keepdim|不需要（自动广播）|必须 keepdim=True（否则无法广播回 4D）|

🔑 卷积层 BN 的核心理解：同一个通道的所有像素（不同位置 \+ 不同样本）共享同一个均值和方差。每个通道独立标准化。

### 1\.4 BatchNorm 类的完整实现

```python
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)           # 全连接：(1, features)
        else:
            shape = (1, num_features, 1, 1)     # 卷积：(1, channels, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))   # γ 初始=1
        self.beta  = nn.Parameter(torch.zeros(shape))  # β 初始=0
        self.moving_mean = torch.zeros(shape)           # 全局均值，初始=0
        self.moving_var  = torch.ones(shape)            # 全局方差，初始=1
    def forward(self, X):
        # 将 moving_mean/var 搬到和 X 同一设备（GPU/CPU 迁移兼容）
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var  = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta,
            self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)
        return Y

```

|参数|初始值|含义|
|---|---|---|
|gamma|1|缩放因子——"可以放大或缩小标准化后的值"|
|beta|0|平移因子——"可以整体偏移"|
|moving\_mean|0|训练过程中积累的全局均值（推理时用）|
|moving\_var|1|训练过程中积累的全局方差（推理时用）|
|momentum|0\.9|指数移动平均的衰减率——越接近 1 越依赖历史值|

### 1\.5 训练 vs 推理的行为切换

```python
if not torch.is_grad_enabled():          # torch.no_grad() 环境
    # 推理模式：用 moving_mean / moving_var（全局统计量）
    X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
else:
    # 训练模式：用当前 batch 的统计量 + 更新 moving 值
    mean = X.mean(...)
    var = ((X - mean) ** 2).mean(...)
    X_hat = (X - mean) / torch.sqrt(var + eps)
    moving_mean = momentum * moving_mean + (1 - momentum) * mean
    moving_var  = momentum * moving_var  + (1 - momentum) * var

```

|模式|使用的均值和方差|是否更新 moving|
|---|---|---|
|训练 (net\.train\(\)\)|当前 batch 的统计量|✅ 更新|
|推理 (net\.eval\(\)\)|全局 moving 统计量|❌ 冻结|

### 1\.6 带 BatchNorm 的 LeNet

```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(2, stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84),     BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10)
)

```

和 Day31 的原始 LeNet 对比：

|位置|原始 LeNet|加 BN 后|
|---|---|---|
|每个卷积层后|Conv → Sigmoid|Conv → BN → Sigmoid|
|每个全连接层后|Linear → Sigmoid|Linear → BN → Sigmoid|
|学习率|0\.9|1\.0（BN 允许更高学习率）|

🎯 BatchNorm 的三大好处在此充分体现：① 学习率从 0\.9 提升到 1\.0——训练更快；② 对参数初始化不再敏感；③ 本身有一定正则化效果（batch 统计量的噪声类似于 Dropout）。

---

## 第二部分：ResNet——残差网络

### 2\.1 核心问题：网络越深，反而越差？

直觉上：56 层网络应该比 20 层网络效果更好（至少不更差——多出来的层可以学"恒等映射"）。但实验结果是：56 层网络在训练集和测试集上的误差都高于 20 层。

这不是过拟合（训练误差也高），而是优化困难——深层网络在 SGD 下很难学到"恒等映射"。

ResNet 的解决思路：与其让网络直接学 $H(x)$（目标映射），不如让它学 $F(x) = H(x) - x$（残差）。如果恒等映射是最优的，只需把 $F(x)$ 推到 0 即可——这比推到恒等映射容易得多。

### 2\.2 残差块（Residual Block）

```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            # 当输入输出形状不匹配时，用 1×1 卷积调整
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))    # Conv → BN → ReLU
        Y = self.bn2(self.conv2(Y))             # Conv → BN（还没加 ReLU）
        if self.conv3:
            X = self.conv3(X)                   # 1×1 卷积调整 X 的形状
        Y += X                                  # ← 核心：残差连接
        return F.relu(Y)                        # 最后过 ReLU
```

### 2\.3 残差块的数据流图解

```text
输入 X
  │
  ├──────────────────────┐
  │                      │
  ▼                      │
Conv3×3 → BN → ReLU      │
  │                      │
  ▼                      │
Conv3×3 → BN              │
  │                      │
  ▼                      │
  Y (未激活)              │
  │                      │
  │    Y += X ◄──────────┘  ← 跳跃连接（skip connection）
  │    (如果形状不匹配，X 先过 1×1 卷积)
  ▼
ReLU → 输出

```

🧠 直觉：跳跃连接给了梯度一条"高速公路"。反向传播时，梯度可以直接通过跳跃连接回流到前面的层（不需要穿过两个卷积层），极大缓解了梯度消失。

### 2\.4 何时需要 use\_1x1conv=True？

残差连接 $Y += X$ 要求 Y 和 X 的形状完全一致。以下两种情况需要 1×1 卷积调整 X：

|情况|原因|use\_1x1conv|strides|
|---|---|---|---|
|输入输出通道数相同，空间不变|直接相加即可|False|1|
|输入输出通道数不同|X 的通道数 ≠ Y 的通道数|True|1|
|空间减半|stride=2，X 的 H×W 和 Y 不同|True|2|

验证：

```python
# 情况 1：通道相同，直接加
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
blk(X).shape  # (4, 3, 6, 6) ← 不变
# 情况 2+3：通道变多 + 空间减半
blk = Residual(3, 6, use_1x1conv=True, strides=2)
blk(X).shape  # (4, 6, 3, 3) ← 通道 3→6，空间 6→3

```

### 2\.5 ResNet\-18 完整结构

以 ResNet\-18 为例（18 层 = 1 个 7×7 conv \+ 4 个 stage × 2 个残差块 × 每块 2 层 conv \+ 1 个 fc）：

```python
# b1: 初始特征提取
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 224→112
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)       # 112→56
)
# 四个残差阶段（每个阶段 2 个残差块 = 4 层卷积）
b2 = resnet_block(64,  64,  2, first_block=True)   # 56→56（不降空间）
b3 = resnet_block(64,  128, 2)                      # 56→28（首块 stride=2）
b4 = resnet_block(128, 256, 2)                      # 28→14
b5 = resnet_block(256, 512, 2)                      # 14→7
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),   # → (512, 1, 1)
                    nn.Flatten(),
                    nn.Linear(512, 10))

```

### 2\.6 resnet\_block 工厂函数

```python
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 块的第一个残差单元：可能改变通道数和空间尺寸
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 后续残差单元：通道和空间不变
            blk.append(Residual(num_channels, num_channels))
    return blk

```

|参数|含义|
|---|---|
|input\_channels|该块第一个残差单元的输入通道数|
|num\_channels|该块所有残差单元的输出通道数|
|num\_residuals|该块的残差单元个数|
|first\_block=True|b2 的第一个块不需要降采样（b1 已做过）|

各块的配置：

|块|input|output|残差单元数|空间变化|
|---|---|---|---|---|
|b2|64|64|2|56×56 → 56×56（首块 first\_block=True，不下采样）|
|b3|64|128|2|56×56 → 28×28（首块 stride=2）|
|b4|128|256|2|28×28 → 14×14|
|b5|256|512|2|14×14 → 7×7|

### 2\.7 ResNet\-18 逐层形状（224×224 输入）

```text
输入:                     (1,   1, 224, 224)
b1: Conv(7×7,s=2,p=3):    (1,  64, 112, 112)   ← (224+6-7)/2+1=112
    MaxPool(3,s=2,p=1):    (1,  64,  56,  56)   ← (112+2-3)/2+1=56
b2: Residual×2(64→64):    (1,  64,  56,  56)    ← 空间不变，通道不变
b3: Residual(64→128,s=2): (1, 128,  28,  28)    ← 首块降采样
    Residual(128→128):    (1, 128,  28,  28)
b4: Residual(128→256,s=2):(1, 256,  14,  14)
    Residual(256→256):    (1, 256,  14,  14)
b5: Residual(256→512,s=2):(1, 512,   7,   7)
    Residual(512→512):    (1, 512,   7,   7)
AdaptiveAvgPool2d((1,1)):  (1, 512,   1,   1)
Flatten:                   (1, 512)
Linear(512, 10):           (1, 10)  → 10 类 logits

```

📐 和 VGG 一样——5 次空间减半（b1 两次 \+ b3/b4/b5 各一次），但 ResNet 用 stride=2 的卷积代替了池化来做降采样（只在 b1 用了一次 MaxPool）。

### 2\.8 ResNet 的设计哲学总结

|设计|做法|为什么|
|---|---|---|
|残差连接|$Y = F(X) + X$|梯度直通，恒等映射易学|
|BatchNorm|每层卷积后加 BN|稳定训练，允许更大学习率|
|瓶颈结构|3×3 → 3×3（没有 1×1 瓶颈）|ResNet\-18/34 用 BasicBlock；ResNet\-50\+ 才用 Bottleneck|
|stride=2 替代池化|块间降采样用 stride=2 的卷积|可学习的降采样|
|AdaptiveAvgPool|最后全局平均池化|没有全连接层堆积（仅 1 个 Linear\(512,10\)）|
|通道翻倍，空间减半|64→128→256→512|和 VGG 一样的哲学|

---

## 第三部分：BatchNorm \+ ResNet = 深层网络的黄金组合

|技术|解决的问题|机制|
|---|---|---|
|BatchNorm|训练不稳定、对 lr 敏感|强制每层输入分布稳定|
|残差连接|深层时梯度消失、恒等映射难学|梯度高速公路 $F(x)+x$|
|两者结合|让 152 层网络也能稳定训练|BN 稳定信号 \+ 残差直通梯度|

🎯 现代 CNN 的标准模板：Conv → BN → ReLU \+ 残差连接。这三样东西从 ResNet 开始成为标配，沿用至今。

---

## 第四部分：经典 CNN 演进全景（Day29\~34 总结）

```text
LeNet (1998)     → 卷积神经网络诞生
    │
AlexNet (2012)   → ReLU + Dropout + MaxPool + GPU
    │
VGG (2014)       → 统一3×3 + 块式设计
    │
NiN (2014)       → 1×1卷积 + 全局池化替代全连接
    │
GoogLeNet (2014) → Inception多尺度并行 + 瓶颈层
    │
                  ┌─ BatchNorm (2015) ← 今天 Day34
                  │
ResNet (2015) ────┘  残差连接 ← 今天 Day34
    │
    ▼
  现代CNN标准模板：Conv → BN → ReLU + 残差连接

```

---

## 第五部分：关键代码段速查

### 5\.1 BatchNorm 自定义层的完整写法

```python
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta  = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var  = torch.ones(shape)
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var  = self.moving_var.to(X.device)
        # ... batch_norm 计算 ...
        return Y

```

### 5\.2 残差块完整写法

```python
class Residual(nn.Module):
    def __init__(self, in_ch, out_ch, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 1, stride=strides) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

```

### 5\.3 ResNet 阶段构造

```python
def resnet_block(in_ch, out_ch, num_res, first_block=False):
    blk = []
    for i in range(num_res):
        if i == 0 and not first_block:
            blk.append(Residual(in_ch, out_ch, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(out_ch, out_ch))
    return blk

```

---

## 第六部分：本日关键记忆点

### BatchNorm

|编号|知识点|一句话|
|---|---|---|
|①|BN 四步|求均值 → 求方差 → 标准化 → γ缩放\+β平移|
|②|γ 和 β|可学习参数——让网络保留"反标准化"的能力|
|③|全连接 vs 卷积 BN|全连接对 feature 求均值（dim=0）；卷积对 batch\+H\+W 求均值（dim=\(0,2,3\)），每个通道独立|
|④|训练 vs 推理|训练用 batch 统计量并更新 moving；推理用 moving 全局统计量|
|⑤|momentum|0\.9——moving 值偏向历史、缓慢更新|
|⑥|BN 三大好处|① 允许更大学习率 ② 降低对初始化的敏感度 ③ 自带轻微正则化|

### ResNet

|编号|知识点|一句话|
|---|---|---|
|⑦|残差学习|$F(x) = H(x) - x$，让网络学"偏离恒等的差值"而非"完整的映射"|
|⑧|跳跃连接|$Y = F(X) + X$——梯度可以直通前面各层|
|⑨|1×1 卷积匹配|当输入输出通道不同或空间减半时，用 1×1 conv 调整 x 的形状|
|⑩|ResNet\-18 结构|b1（stem）→ b2/b3/b4/b5（各 2 个残差块）→ GlobalAvgPool → Linear|
|⑪|BN \+ 残差|现代 CNN 标配：Conv → BN → ReLU \+ skip connection|
|⑫|无全连接堆积|ResNet 只有一个 Linear(512,10\)，比 AlexNet/VGG 的 3 层 FC 参数量少得多|
