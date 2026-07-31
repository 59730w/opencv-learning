# Day31：汇聚层/池化层 \+ LeNet——第一个完整CNN

**核心主题**：最大池化/平均池化 → 池化的填充和步幅 → 多通道池化 → LeNet 结构逐层解析 → GPU 训练 CNN

---

## 第一部分：汇聚层/池化层

### 1\.1 池化层要解决什么问题？

卷积层可以有效提取图像特征，但存在两个致命缺陷：

|问题|说明|
|---|---|
|**特征图仍太大**|28×28 的图像经过 Conv2d(1,6,5,padding=2\) 后尺寸保持28×28，计算量无法降低，深层网络压力极大|
|**对位置太敏感**|卷积核精准记录特征位置，同一边缘特征仅平移1个像素，输出特征图就会完全改变，泛化能力差|

**池化层两大核心作用**：

① **降采样**：缩小特征图空间尺寸，大幅减少后续网络层计算量与参数量

② **引入平移不变性**：图像微小平移、偏移后，池化输出基本保持不变，提升模型泛化能力

### 1\.2 手写池化函数

```python
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                # 最大池化：取窗口内最大值
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                # 平均池化：取窗口内平均值
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

```

与 Day29 手写卷积函数 corr2d 对比：

|操作|corr2d（卷积）|pool2d（池化）|
|---|---|---|
|窗口处理逻辑|逐元素相乘后求和|直接取窗口 max 或 mean|
|可学习参数|✅ 有（卷积核权重、偏置）|❌ 无任何可学习参数|
|输出尺寸公式|H\_out = H \- h \+ 1|完全一致|

### 1\.3 最大池化 vs 平均池化

```python
X = torch.tensor([[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]])

print(pool2d(X, (2, 2), 'max'))  # 输出：[[4., 5.], [7., 8.]]
print(pool2d(X, (2, 2), 'avg'))  # 输出：[[2., 3.], [5., 6.]]

```

|池化类型|核心原理|直观理解|PyTorch 对应类|
|---|---|---|---|
|最大池化|提取窗口内像素最大值|筛选区域内最显著、最重要的特征|nn\.MaxPool2d|
|平均池化|计算窗口内像素平均值|统计区域内特征的整体响应水平|nn\.AvgPool2d|

💡 行业现状：现代 CNN 基本全部使用**最大池化**；LeNet 使用平均池化仅为历史设计原因。

### 1\.4 池化层的填充和步幅

池化层与卷积层**完全共用同一套输出尺寸公式**：

H\_out = (H \+ 2 \* p \- k\) // s \+ 1

W\_out = (W \+ 2 \* p \- k\) // s \+ 1

参数说明：p=填充、k=窗口尺寸、s=步幅、// 为向下取整

**实验1：默认步幅（无重叠池化）**

```python
X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
pool2d = nn.MaxPool2d(3)  # kernel=3，默认 stride=kernel_size=3
# 尺寸计算：(4-3)//3 + 1 = 1
# 输出尺寸：1×1

```

核心规则：**nn\.MaxPool2d 默认 stride = kernel\_size**，窗口无重叠，是最常用池化配置。

**实验2：Padding \+ Stride 组合使用**

```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# 输入4×4，填充后6×6，3×3窗口、步幅2
# 输出尺寸：2×2

```

**实验3：非正方形池化窗口、非对称参数**

```python
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
# 高度方向：kernel=2、padding=0、stride=2
# 宽度方向：kernel=3、padding=1、stride=3

```

### 1\.5 多通道池化——通道数永久不变

```python
# 单通道转为双通道
X = torch.cat((X, X + 1), 1)   # shape: (1,1,4,4) → (1,2,4,4)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X).shape)         # 输出：torch.Size([1, 2, 2, 2])

```

核心特性：池化操作**逐通道独立执行**，通道之间互不干扰

|对比项|卷积层|池化层|
|---|---|---|
|通道数变化|可自由修改（out\_channels 控制）|输入输出通道数严格相等|
|通道运算逻辑|多输入通道融合计算单输出通道|每个通道独立池化，无通道融合|
|参数量|存在大量可学习参数|参数永久为0，纯固定运算|

总结：池化只改变特征图高度、宽度，**绝对不改变通道数**。

### 1\.6 池化 vs stride=2 卷积（降采样方式对比）

现代CNN两大主流降采样方案对比：

|特性|MaxPool2d(2, stride=2\)|Conv2d(\.\.\., stride=2\)|
|---|---|---|
|参数量|0，无参数|存在可学习参数|
|降采样逻辑|固定取最大值，硬降采样|模型自适应学习降采样规则|
|计算量|极低，运算速度快|较高，需要参数运算|
|平移不变性|强，抗微小位移干扰|较弱|
|适用场景|VGG、早期ResNet|现代ResNet、ViT主流方案|

---

## 第二部分：LeNet——第一个完整CNN

### 2\.1 历史意义

LeNet 由 Yann LeCun 于1998年提出，专为MNIST手写数字识别设计，是**人类首个成功落地的卷积神经网络**。

奠定了沿用至今的CNN经典骨架：

**卷积 → 激活 → 池化 → 卷积 → 激活 → 池化 → 全连接 → 输出**

### 2\.2 LeNet 完整结构代码

```python
import torch
import torch.nn as nn

net = nn.Sequential(
    # 第一个卷积块：特征提取
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # 第二个卷积块：深层特征提取
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    # 分类器全连接层
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

```

### 2\.3 逐层形状流转（核心重点）

输入：28×28 Fashion\-MNIST 单通道灰度图，逐层维度变化：

```python
X = torch.rand(1, 1, 28, 28)
for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__:12s} → {X.shape}")

```

|层级|网络层|核心操作|输出形状|尺寸计算|
|---|---|---|---|---|
|输入|\-|原始灰度图|(1, 1, 28, 28\)|\-|
|1|Conv2d|1通道→6通道，5×5核，padding=2|(1, 6, 28, 28\)|(28\+4\-5\)//1\+1=28|
|\-|Sigmoid|非线性激活|(1, 6, 28, 28\)|尺寸不变|
|2|AvgPool2d|2×2池化，stride=2|(1, 6, 14, 14\)|(28\-2\)//2\+1=14|
|3|Conv2d|6通道→16通道，5×5核，无padding|(1, 16, 10, 10\)|(14\-5\)//1\+1=10|
|\-|Sigmoid|非线性激活|(1, 16, 10, 10\)|尺寸不变|
|4|AvgPool2d|2×2池化，stride=2|(1, 16, 5, 5\)|(10\-2\)//2\+1=5|
|5|Flatten|维度展平|(1, 400\)|16×5×5=400|
|6|Linear|400→120|(1, 120\)|\-|
|7|Linear|120→84|(1, 84\)|\-|
|8|Linear|84→10（分类输出）|(1, 10\)|10个类别Logits|

### 2\.4 形状变化可视化流程

(1,1,28,28\) ——Conv2d\+Sigmoid——\> (1,6,28,28\) ——AvgPool——\> (1,6,14,14\)

(1,6,14,14\) ——Conv2d\+Sigmoid——\> (1,16,10,10\) ——AvgPool——\> (1,16,5,5\)

(1,16,5,5\) ——Flatten——\> (1,400\)

(1,400\) ——Linear(400,120\)——\> (1,120\)

(1,120\) ——Linear(120,84\)——\> (1,84\)

(1,84\) ——Linear(84,10\)——\> (1,10\) 最终分类输出

### 2\.5 LeNet 核心设计要点

|设计细节|LeNet 实现方式|设计目的|
|---|---|---|
|首层卷积填充|Conv2d(1,6,5,padding=2\)|保证卷积后尺寸不变，保留图像边缘特征|
|二层卷积无填充|Conv2d(6,16,5\)|主动缩小特征图尺寸，压缩冗余信息|
|池化步幅配置|所有池化层 stride=2|每次池化空间尺寸减半，逐级降采样|
|激活函数选择|全程使用 Sigmoid|历史设计选择，现代网络统一替换为 ReLU|
|全连接输入维度|16×5×5=400|由前置卷积、池化层尺寸逐级推导而来，层层绑定|
|整体网络结构|2次卷积池化 \+ 3层全连接|建立CNN通用骨架：特征提取 \+ 特征分类|

### 2\.6 GPU 训练核心函数 train\_ch6

适配GPU的完整CNN训练函数，解决CPU训练速度慢的问题：

```python
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    # 权重初始化：Xavier初始化，适配卷积/全连接层
    net.apply(lambda m: nn.init.xavier_uniform_(m.weight)
              if isinstance(m, (nn.Linear, nn.Conv2d)) else None)
    # 模型迁移至GPU/CPU
    net.to(device)

    # 优化器与损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # 训练主循环
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            # 核心：每批次数据迁移至对应设备
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        
        # 每轮结束评估测试集准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)

```

CPU版\(train\_ch3\)与GPU版\(train\_ch6\)对比：

|对比项目|train\_ch3（CPU版）|train\_ch6（GPU版）|
|---|---|---|
|运行设备|默认CPU运行|自动适配GPU/CPU|
|数据处理|无需设备迁移|每批次X、y必须迁移至模型设备|
|评估函数|evaluate\_accuracy|evaluate\_accuracy\_gpu|
|参数初始化|手动正态初始化std=0\.01|Xavier均匀初始化，更稳定|

### 2\.7 设备自适应评估函数 evaluate\_accuracy\_gpu

```python
def evaluate_accuracy_gpu(net, data_iter, device=None):
    net.eval()
    # 自动从模型参数推断运行设备，无需手动传入
    if not device:
        device = next(iter(net.parameters())).device
    
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            # 数据与模型保持同一设备
            X, y = X.to(device), y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

```

核心亮点：`next(iter(net.parameters())).device` 自动获取模型所在设备，代码通用性极强。

### 2\.8 训练效果

超参数配置：epoch=10、学习率lr=0\.9

LeNet 在 Fashion\-MNIST 数据集上测试准确率可达 **82%左右**；配合更多迭代轮数、数据增广，准确率可进一步提升。

---

## 第三部分：CNN vs MLP——模型性能质变

|对比维度|Day22 MLP 全连接网络|Day31 LeNet 卷积网络|
|---|---|---|
|网络类型|纯全连接结构|卷积\+池化\+全连接混合结构|
|首层参数量|784×256\+256 = 200960|1×6×5×5\+6 = 156|
|总参数量|约200K|约62K，参数大幅精简|
|图像理解方式|将像素视为独立数值，丢失空间信息|保留2D空间结构，提取局部特征|
|平移不变性|❌ 像素微小位移，结果完全改变|✅ 池化带来天然平移容忍性|
|训练速度|参数量大，训练慢|参数精简，收敛更快|

---

## 第四部分：代码优化改进点

原生LeNet为历史经典结构，现代应用可优化两点：

|模块|原生LeNet设计|现代优化方案|优化原因|
|---|---|---|---|
|激活函数|Sigmoid|ReLU|缓解梯度消失，收敛速度更快|
|池化方式|AvgPool2d 平均池化|MaxPool2d 最大池化|聚焦显著特征，适配现代数据集|

注：优化仅用于实战开发，学习LeNet原理需保留原始结构。

---

## 第五部分：CNN基础体系全景回顾（Day29\-Day31）

Day29：互相关运算 → 手写卷积层 → 边缘检测 → 自动学习卷积核

Day30：填充与步幅 → 多输入多输出通道 → 1×1卷积通道变换

Day31：最大/平均池化 → LeNet完整网络组装 → GPU高速训练CNN

✅ 三日学习完成**CNN全套基础构件**，掌握经典卷积网络搭建、训练、调优核心逻辑

---

## 第六部分：本日核心记忆点

### 池化层核心知识点

|编号|知识点|一句话总结|
|---|---|---|
|①|池化两大核心作用|降采样压缩特征图尺寸、引入平移不变性提升泛化能力|
|②|最大/平均池化区别|最大池化取显著特征，平均池化取区域均值，现代CNN优先用最大池化|
|③|池化无参数|池化是固定数学运算，无任何可学习权重与偏置|
|④|池化通道特性|仅改变特征图高宽，输入输出通道数完全一致，逐通道独立运算|
|⑤|池化默认步幅|MaxPool2d默认stride=kernel\_size，实现无重叠池化|

### LeNet 核心知识点

|编号|知识点|一句话总结|
|---|---|---|
|⑥|LeNet三段式结构|卷积池化×2 \+ 全连接×3，是所有现代CNN的原始模板|
|⑦|400维特征来源|图像经两次卷积池化得到16×5×5特征，展平为400维向量输入全连接层|
|⑧|GPU训练标配流程|模型移至device、每批次数据同步迁移、设备自适应评估|
|⑨|卷积参数优势|LeNet仅62K参数，远少于MLP的200K，参数效率碾压全连接网络|
|⑩|Xavier初始化|替代手动正态初始化，让CNN训练初始权重更合理，收敛更稳定|
