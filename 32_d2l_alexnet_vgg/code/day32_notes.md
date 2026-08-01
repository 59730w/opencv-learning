# Day32：AlexNet \+ VGG——深度卷积神经网络

**核心主题**：AlexNet——更大的核、更深的网络、Dropout防过拟合 → VGG——块式设计、统一3×3卷积、深度与规整的典范

---

## 第一部分：AlexNet

### 1\.1 历史地位

2012 年 ImageNet 竞赛冠军，将**深度卷积神经网络**首次推入公众视野。AlexNet 证明了：只要**足够深 \+ 足够多数据 \+ GPU**，CNN 就能做到此前无法想象的图像识别水平。

LeNet（1998）→ 间隔 14 年 → AlexNet（2012）：14 年间两大瓶颈被彻底打破——**GPU 算力**和**海量标注数据（ImageNet）**。

### 1\.2 AlexNet vs LeNet 核心差异

|维度|LeNet \(1998\)|AlexNet \(2012\)|
|---|---|---|
|输入尺寸|28×28|**224×224**|
|第一层卷积核|5×5|**11×11**（更大感受野）|
|首层通道数|1→6|1→**96**（16倍扩容）|
|全连接层维度|120, 84|**4096, 4096**（数十倍提升）|
|激活函数|Sigmoid|**ReLU**|
|正则化策略|无|**Dropout\(0\.5\)**|
|池化方式|AvgPool 平均池化|**MaxPool 最大池化**|
|训练硬件|单核 CPU|**双 GPU 并行训练**|

### 1\.3 AlexNet 完整结构逐层解析

```python
import torch
import torch.nn as nn

net = nn.Sequential(
    # ===== 第一个卷积块 =====
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # ===== 第二个卷积块 =====
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # ===== 连续三层卷积（无中间池化，加深特征）=====
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),   # 三连卷积后统一池化降维
    # ===== 分类器头部 =====
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.5),   # Dropout 抑制过拟合
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096, 10)
)

```

### 1\.4 逐层形状流转（输入 224×224）

输入尺寸：(1, 1, 224, 224\)

**第一层卷积\+激活\+池化**

Conv2d(1,96,11,stride=4,padding=1\) \+ ReLU → 输出：(1, 96, 54, 54\)

尺寸计算：(224 \+ 2\*1 \- 11\) // 4 \+ 1 = 54

MaxPool2d(3,stride=2\) → 输出：(1, 96, 26, 26\)

尺寸计算：(54 \- 3\) // 2 \+ 1 = 26

**第二层卷积\+激活\+池化**

Conv2d(96,256,5,padding=2\) \+ ReLU → 输出：(1, 256, 26, 26\)

尺寸计算：padding=2、kernel=5，输入输出尺寸一致

MaxPool2d(3,stride=2\) → 输出：(1, 256, 12, 12\)

尺寸计算：(26 \- 3\) // 2 \+ 1 = 12

**三层连续卷积（无中间池化）**

Conv2d(256→384,3,padding=1\) → 输出：(1, 384, 12, 12\) 尺寸不变

Conv2d(384→384,3,padding=1\) → 输出：(1, 384, 12, 12\) 持续深化特征

Conv2d(384→256,3,padding=1\) → 输出：(1, 256, 12, 12\) 收窄通道

**末次池化降维**

MaxPool2d(3,stride=2\) → 输出：(1, 256, 5, 5\)

尺寸计算：(12 \- 3\) // 2 \+ 1 = 5

**展平\+全连接分类**

Flatten → (1, 6400\)  256×5×5 = 6400

Linear(6400,4096\) \+ ReLU \+ Dropout → (1, 4096\)

Linear(4096,4096\) \+ ReLU \+ Dropout → (1, 4096\)

Linear(4096,10\) → (1, 10\) 输出10类分类概率

### 1\.5 关键设计解读

|设计方案|AlexNet 具体做法|设计初衷|
|---|---|---|
|大核大步幅开局|首层11×11卷积核，stride=4大步幅|快速压缩超大输入图尺寸：224→54→26→12→5，降低计算压力|
|无池化连续卷积|三层3×3卷积堆叠，中间不插入池化层|验证**网络深度优先级高于宽度**，用小核堆叠替代单一超大核|
|重叠池化设计|MaxPool\(3, stride=2\)，窗口大于步幅|池化窗口相互重叠，相比无重叠池化，小幅提升特征提取精度|
|Dropout正则化|两层4096维全连接后均添加Dropout\(0\.5\)|全连接层参数量高达26M，随机丢弃神经元，彻底解决严重过拟合问题|
|固定特征维度|最终特征图 256×5×5=6400|和LeNet 16×5×5=400逻辑一致，前置网络尺寸固定全连接输入维度|

### 1\.6 训练配置

```python
import d2l.torch as d2l

# 超参数设置
batch_size = 128
lr, num_epochs = 0.01, 10

# 数据集预处理：将28×28原图resize至224×224适配AlexNet输入
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# GPU训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

```

---

## 第二部分：VGG

### 2\.1 设计哲学：极致规整模块化

AlexNet各层配置杂乱无章，11×11、5×5、3×3卷积核混用，属于人工定制的网络结构。

**VGG核心思想**：抛弃杂乱定制，采用**统一3×3卷积核 \+ 块式堆叠**，让**网络深度**成为唯一变量，结构极简、规整、可扩展。

VGG解答核心问题：在规整统一的设计范式下，网络越深，特征提取能力越强、效果越好。

### 2\.2 VGG 块（vgg\_block）——模块化经典范本

```python
def vgg_block(num_convs, in_channels, out_channels):
    """
    VGG基础模块
    :param num_convs: 当前块堆叠的卷积层数
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :return: 卷积+激活+池化的序列化模块
    规则：所有卷积固定3×3、padding=1；池化固定2×2、stride=2
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels  # 块内通道数保持统一
    # 块末尾统一池化，空间尺寸减半
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

```

|模块属性|固定配置|设计原因|
|---|---|---|
|卷积核尺寸|全局固定 3×3|堆叠小核替代大核，更少参数、更多非线性激活|
|填充 padding|固定 padding=1|3×3卷积\+padding=1，卷积前后尺寸完全不变|
|池化参数|2×2窗口，stride=2|每个模块结束空间尺寸严格减半，规则统一|
|通道变化规则|块内通道不变，逐块通道翻倍|空间尺寸减半、通道数翻倍，整体计算量动态平衡|

### 2\.3 VGG\-11 网络配置

```python
# VGG-11 卷积块配置：(卷积层数, 输出通道数)
conv_arch = ((1, 64),      # 块1：1层卷积，输出64通道
             (1, 128),     # 块2：1层卷积，输出128通道
             (2, 256),     # 块3：2层卷积，输出256通道
             (2, 512),     # 块4：2层卷积，输出512通道
             (2, 512))     # 块5：2层卷积，输出512通道

# 总层数计算：8层卷积 + 3层全连接 = 11层，即 VGG-11

```

### 2\.4 VGG 网络工厂函数

```python
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1  # 输入单通道灰度图
    # 堆叠所有VGG卷积块
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    # 卷积特征提取 + 全连接分类头
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 224经过5次池化减半：224 / 2^5 = 7，最终特征图 512×7×7
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

```

### 2\.5 VGG\-11 形状流转（224×224输入）

初始输入：(1, 1, 224, 224\)

块1（1×Conv\+MaxPool）→ (1, 64, 112, 112\)  224/2=112

块2（1×Conv\+MaxPool）→ (1, 128, 56, 56\)    112/2=56

块3（2×Conv\+MaxPool）→ (1, 256, 28, 28\)    56/2=28

块4（2×Conv\+MaxPool）→ (1, 512, 14, 14\)    28/2=14

块5（2×Conv\+MaxPool）→ (1, 512, 7, 7\)      14/2=7

Flatten 展平 → (1, 25088\)  512×7×7=25088

全连接层：25088→4096→4096→10

### 2\.6 核心设计：两个3×3卷积替代一个5×5卷积

**感受野等价原理**：两层3×3卷积堆叠，等效单层5×5卷积的感受野

第一层3×3卷积提取局部特征，第二层3×3卷积基于局部特征再次卷积，最终单个像素可感知原始图像5×5区域范围。

**参数量对比（输入64通道、输出128通道）**

单层5×5卷积参数：64 × 128 × 5 × 5 = 204800

双层3×3卷积总参数：64×128×3×3 \+ 128×128×3×3 = 73728 \+ 147456 = 221184

**核心优势**：参数量基本持平，但双层卷积多一次ReLU非线性激活，**模型特征表达能力更强**，这是VGG最核心的设计洞察。

### 2\.7 教学优化：通道数缩减

原版VGG通道数过大，训练显存占用极高，针对Fashion\-MNIST小数据集教学，采用比例缩减策略：

```python
ratio = 4
# 所有通道数除以4，轻量化模型、降低训练成本
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# 缩减后配置：64→16, 128→32, 256→64, 512→128, 512→128

```

注：真实项目、大数据集训练时，取消通道缩减，使用原版配置即可。

### 2\.8 手写训练循环与自定义绘图

VGG训练放弃d2l封装函数，采用原生手写训练循环，完全自定义绘图、保存图片，自由度更高：

```python
import matplotlib.pyplot as plt

# 记录训练指标
train_loss_list = []
train_acc_list  = []
test_acc_list   = []
epochs = list(range(num_epochs))

# 手动训练循环
for epoch in range(num_epochs):
    # 训练过程省略
    train_loss_list.append(train_l)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

# 自定义绘制三曲线并高清保存
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(epochs, train_loss_list, label="train loss")
ax.plot(epochs, train_acc_list,  label="train acc")
ax.plot(epochs, test_acc_list,   label="test acc")
ax.legend()
plt.savefig("vgg_manual_curve.png", dpi=300, bbox_inches="tight")

```

---

## 第三部分：AlexNet vs VGG 架构哲学对比

|对比维度|AlexNet|VGG|
|---|---|---|
|卷积核规格|11×11、5×5、3×3 多尺寸混用|全局统一 3×3 卷积核|
|网络结构|平铺逐层手动配置，无模块化|块式模块化设计，工厂函数生成网络|
|设计原则|经验直觉驱动，每层单独定制|规整统一驱动，仅通过深度提升性能|
|网络深度|固定8层结构|可配置11/13/16/19层，扩展性极强|
|通道变化策略|96→256→384→384→256，无固定规律|64→128→256→512→512，空间减半、通道翻倍|
|代码复用性|低，结构固定不可修改|极高，修改arch参数即可切换网络深度|
|正则化|Dropout(0\.5\)|Dropout(0\.5\)|

**进化本质**：从AlexNet人工调参的「定制艺术品」，进化为VGG标准化、可复用、可扩展的「通用模块化框架」，奠定了后续ResNet、DenseNet等所有经典CNN的设计范式。

---

## 第四部分：三代CNN全连接输入尺寸溯源

LeNet → AlexNet → VGG 全连接层输入维度推导逻辑：

|网络模型|最终特征图尺寸|展平维度|尺寸推导逻辑|
|---|---|---|---|
|LeNet|16×5×5|400|28→卷积池化→14→卷积池化→5|
|AlexNet|256×5×5|6400|224→大核大步幅压缩→多级池化最终得到5×5特征图|
|VGG|512×7×7|25088|224经过5次2倍池化，224/2^5=7，固定输出7×7特征图|

核心能力：通过网络层级配置，反向推导特征图尺寸，是调试CNN模型、解决维度报错的核心基本功。

---

## 第五部分：LeNet→AlexNet→VGG 演进全景

**LeNet (1998\)**

Conv(1→6,5\) → AvgPool → Conv(6→16,5\) → AvgPool

FC(400,120\) → FC(120,84\) → FC(84,10\)

激活：Sigmoid \| 正则：无 \| 池化：平均池化

**AlexNet (2012\)**

Conv(1→96,11,s=4\) → MaxPool → Conv(96→256,5,p=2\) → MaxPool

Conv(256→384→384→256,3\) → MaxPool

FC(6400,4096\)Dropout → FC(4096,4096\)Dropout → FC(4096,10\)

激活：ReLU \| 正则：Dropout(0\.5\) \| 池化：重叠最大池化

**VGG (2014\)**

块1(Conv×1\)→Pool → 块2(Conv×1\)→Pool → 块3(Conv×2\)→Pool

块4(Conv×2\)→Pool → 块5(Conv×2\)→Pool

FC(25088,4096\)Dropout → FC(4096,4096\)Dropout → FC(4096,10\)

激活：ReLU \| 正则：Dropout(0\.5\) \| 池化：标准2×2最大池化

---

## 第六部分：本日核心记忆点

### AlexNet 核心知识点

|编号|知识点|一句话总结|
|---|---|---|
|①|大核大步幅开局|11×11卷积\+stride=4，对224大图激进压缩，快速降维|
|②|无池化三连卷积|中间三层卷积堆叠无池化，证明网络深度比浅层池化更重要|
|③|重叠池化机制|MaxPool(3,stride=2\)窗口重叠滑动，小幅提升特征精度|
|④|Dropout防过拟合|超大参数量全连接层搭配0\.5丢弃率，是AlexNet收敛的关键|
|⑤|输入尺寸适配|Fashion\-MNIST原图28×28必须resize至224×224才能适配网络|

### VGG 核心知识点

|编号|知识点|一句话总结|
|---|---|---|
|⑥|vgg\_block模块|固定卷积\+激活\+池化结构，是CNN模块化设计的经典模板|
|⑦|统一3×3卷积|双层3×3堆叠等效5×5感受野，参数持平、更多非线性、表达更强|
|⑧|尺度平衡策略|空间尺寸逐块减半、通道数逐块翻倍，整体计算量均衡稳定|
|⑨|参数化网络结构|修改conv\_arch元组即可切换VGG11/16/19，代码复用性拉满|
|⑩|通道缩减技巧|教学场景用ratio缩减通道，降低显存压力，实战恢复原版配置|
|⑪|手写训练绘图|脱离框架封装，自定义训练循环与高清绘图，完全可控可定制|
