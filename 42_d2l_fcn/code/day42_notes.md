# Day42：全卷积网络 FCN

**核心主题**：FCN 架构设计（全卷积替代全连接）→ 双线性上采样核 → ResNet\-18 作为 backbone → 转置卷积上采样 → 逐像素分类训练与可视化

---

# 一、FCN 是什么？

## 1\.1 核心思想

**FCN（Fully Convolutional Network，全卷积网络）**：把图像分类网络（如 ResNet）的**全连接层全部替换成卷积层**，从而输出**每个像素的类别预测**，而不是整张图的一个类别标签。

|对比|图像分类网络|FCN（全卷积网络）|
|---|---|---|
|输出|1 个类别标签（整张图）|**逐像素类别图**（H×W）|
|末尾结构|全连接层|**卷积层 \+ 转置卷积**|
|输入尺寸限制|固定（FC 层要求）|**任意尺寸**（全卷积无限制）|
|用途|分类|**语义分割**|

图像分类：  图片 → 卷积 → 展平 → 全连接 → [类别标签\]

FCN：       图片 → 卷积 → 1×1卷积 → 转置卷积上采样 → [逐像素类别图\]

🧠 FCN 的洞察：全连接层本质上是「每个位置权重不同」的卷积。把它换成 1×1 卷积 \+ 转置卷积，网络就能输出与输入同尺寸的**密集预测**。

## 1\.2 FCN 的完整架构

输入图像 (3, 320, 480\)

│

▼

ResNet\-18（去掉最后两层：全局池化 \+ 全连接）

│  输出特征图 (512, 10, 15\)

▼

1×1 卷积（512 → 21 类）

│  输出 (21, 10, 15\) —— 每个位置是 21 类的分数

▼

转置卷积（64×64 核，stride=32，padding=16）上采样 ×32

│  输出 (21, 320, 480\) —— 恢复到原图尺寸

▼

逐像素 argmax → 每个像素的类别 (320, 480\)

---

# 二、搭建 FCN（代码核心）

## 2\.1 加载预训练 ResNet\-18 作为 backbone

```python
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]

```

pretrained\_net\.children\(\) 最后三层说明：

|层级|内容|作用|
|---|---|---|
|\-3|GlobalAvgPool|全局平均池化|
|\-2|Flatten|展平特征图|
|\-1|Linear\(512, 1000\)|ImageNet 1000分类头|

## 2\.2 截断网络：去掉全局池化和全连接

```python
net = nn.Sequential(*list(pretrained_net.children())[:-2])
# 去掉最后两层（GlobalAvgPool 和 Flatten），保留卷积特征提取部分
X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)   # torch.Size([1, 512, 10, 15])

```

|参数|值|说明|
|---|---|---|
|输入|(1, 3, 320, 480\)|1 张 320×480 的 RGB 图|
|输出|(1, 512, 10, 15\)|512 通道特征图，空间缩小 32 倍|

320/32 = 10，480/32 = 15。ResNet\-18 经过 5 次 stride=2 下采样，空间缩放倍数：$2^5=32$ 倍。

## 2\.3 添加 FCN 头部：1×1 卷积 \+ 转置卷积

```python
num_classes = 21    # VOC 的 21 类
# ① 1×1 卷积：把 512 通道压缩到 21 类
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
# ② 转置卷积：上采样 32 倍恢复到原图尺寸
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))

```

|组件|参数|作用|
|---|---|---|
|final\_conv|Conv2d(512, 21, 1\)|每个位置输出 21 类的分类分数|
|transpose\_conv|ConvTranspose2d(21, 21, 64, stride=32, padding=16\)|上采样 32 倍复原原图尺寸|

### 转置卷积输出尺寸验证（Day41 公式）

转置卷积尺寸公式：$H_{out} = (H-1) \times s - 2p + k$

$H_{out} = (10-1) \times 32 - 2 \times 16 + 64 = 288 - 32 + 64 = 320$

$W_{out} = (15-1) \times 32 - 2 \times 16 + 64 = 448 - 32 + 64 = 480$

最终输出**\(21, 320, 480\)**，完美匹配输入原图尺寸 ✅

🎯 参数设计原因：kernel=64, stride=32, padding=16 可精准抵消 ResNet 的32倍下采样，一次性复原原图尺寸。

---

# 三、双线性插值核

## 3\.1 为什么要初始化转置卷积核？

转置卷积是可学习参数，但随机初始化会导致训练初期上采样效果混乱、收敛速度慢。FCN 核心优化：**用双线性插值核初始化转置卷积**，让网络初始状态即可完成平滑上采样，再通过训练微调参数。

回顾：转置卷积本质是输入元素扩散填充输出区域，使用双线性核可实现无锯齿、平滑的插值上采样效果。

## 3\.2 双线性核的数学构造

```python
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2          # 缩放因子
    if kernel_size % 2 == 1:
        center = factor - 1                  # 奇数核：中心坐标
    else:
        center = factor - 0.5                # 偶数核：中心坐标
    # 生成网格坐标
    og = (torch.arange(kernel_size).reshape(-1, 1),    
          torch.arange(kernel_size).reshape(1, -1))    
    # 双线性权重：距中心越近权重越大，线性衰减
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    # 通道权重赋值
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

```

核心直觉：k×k权重矩阵呈**金字塔分布**，中心权重最大，向四周线性衰减至0，可实现平滑图像放大。

### 双线性核验证示例

```python
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
img = ToTensor()(Image.open("catdog.jpg"))     # (3, H, W)
Y = conv_trans(img.unsqueeze(0))               # 上采样 2 倍
out_img = Y[0].permute(1, 2, 0).detach()

```

|参数|值|说明|
|---|---|---|
|kernel\_size=4|4×4 卷积核|适配2倍上采样因子|
|padding=1, stride=2|步长2、填充1|输出尺寸为输入2倍|
|bias=False|无偏置|纯双线性插值效果，无偏移|

## 3\.3 初始化 FCN 的转置卷积

```python
W = bilinear_kernel(num_classes, num_classes, 64)   # 21类通道 64×64双线性核
net.transpose_conv.weight.data.copy_(W)              # 赋值初始化转置卷积

```

所有类别通道共享同一双线性核，网络初始具备平滑上采样能力，后续训练自适应微调权重。

---

# 四、FCN 的损失函数

## 4\.1 逐像素交叉熵

```python
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

```

|步骤|说明|
|---|---|
|F\.cross\_entropy(\.\.\., reduction='none'\)|逐像素计算交叉熵，输出形状 (B, H, W\)|
|\.mean(1\)\.mean(1\)|对高、宽维度求平均，得到每个样本的平均像素损失 (B,\)|

🔑 核心逻辑：将图像**每一个像素视为独立分类样本**，对所有像素的交叉熵损失取均值，作为全局损失。

---

# 五、训练

## 5\.1 训练配置

```python
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)   # 复用Day41 VOC数据集
num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

```

|超参数|值|说明|
|---|---|---|
|batch\_size|32|大图训练，显存占用较高|
|crop\_size|\(320, 480\)|VOC数据集统一裁剪尺寸|
|num\_epochs|5|数据集体量较大，5轮可观察收敛趋势|
|lr|0\.001|SGD小学习率，稳定微调预训练权重|
|wd|1e\-3|权重衰减，防止过拟合|
|损失函数|逐像素交叉熵|适配语义分割逐像素分类任务|

⚠️ 显存适配提示：320×480大图\+batch=32对显存要求较高，4GB显存设备可适当减小batch\_size或crop\_size。

---

# 六、预测与可视化

## 6\.1 预测函数

```python
def predict(img, net, test_iter, devices):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)   # ① 归一化 + 新增batch维度
    pred = net(X.to(devices[0])).argmax(dim=1)                # ② 前向推理+逐像素分类
    return pred.reshape(pred.shape[1], pred.shape[2])          # ③ 去除batch维度，输出(H,W)

```

|步骤|说明|
|---|---|
|normalize\_image|复用VOC数据集标准化规则（归一化\+ImageNet均值方差）|
|unsqueeze(0\)|补充batch维度，适配模型输入格式 (1,3,H,W\)|
|argmax(dim=1\)|沿类别维度取最大值，得到逐像素类别索引|
|reshape|去除batch维度，得到纯类别索引图 (H,W\)|

## 6\.2 类别索引 → 彩色图

```python
def label2image(pred, devices):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])   # 加载21类颜色映射表
    X = pred.long()                                                # 转为整型索引
    return colormap[X, :]                                          # 索引映射为彩色图 (H,W,3)

```

转换流程：**预测索引图\(H,W\) → 查表VOC\_COLORMAP → 彩色分割图\(H,W,3\)**，实现分割结果可视化。

## 6\.3 可视化对比代码

```python
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)  # 原图裁剪
    pred = label2image(predict(X, net, test_iter, devices), devices)        # 预测分割图
    imgs += [X.permute(1, 2, 0), pred.cpu(),
             torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]
# 三组图像：原图、模型预测分割图、真实标签图
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)

```

可视化排版：

第1行：原图

第2行：FCN 预测分割图

第3行：真实标签图

通过色彩相似度直观判断分割精度，色彩越接近，分割效果越好。

---

# 七、FCN 完整架构总结

┌─────────────────────────────────────────────────────┐

│ FCN 架构                                             │

│                                                     │

│ 输入 (3, 320, 480\)                                  │

│   │                                                 │

│   ▼                                                 │

│ ResNet\-18（去掉池化\+FC）                            │

│   │ 特征图 (512, 10, 15\)  ← 32 倍下采样             │

│   ▼                                                 │

│ 1×1 卷积 (512 → 21\)                                │

│   │ 类别分数图 (21, 10, 15\)                         │

│   ▼                                                 │

│ 转置卷积 (64核, stride=32, padding=16\)  双线性核初始化 │

│   │ 上采样 32 倍 → (21, 320, 480\)                   │

│   ▼                                                 │

│ 逐像素 argmax → 类别图 (320, 480\)                   │

│   ▼                                                 │

│ 查 VOC\_COLORMAP → 彩色分割图                        │

└─────────────────────────────────────────────────────┘

---

# 八、与之前课程的联系

|往期课程|FCN 中的具体体现|
|---|---|
|Day41 转置卷积|FCN核心上采样模块，64核32步长转置卷积完成32倍上采样|
|Day41 语义分割数据集|复用VOC数据集加载、归一化、颜色映射全套逻辑|
|Day34 ResNet|ResNet\-18作为主干网络，截断末端分类层用于特征提取|
|Day34 微调|使用ImageNet预训练权重迁移学习，加速收敛|
|Day32 AlexNet大核|采用64×64大卷积核实现全局一次性上采样|
|Day26 预训练加载|pretrained=True 快速加载开源预训练权重|
|Day21 交叉熵|将单样本交叉熵扩展为逐像素交叉熵损失|

---

# 九、FCN vs 后续 U\-Net

为U\-Net实战做铺垫，核心差异对比如下：

|对比维度|FCN|U\-Net|
|---|---|---|
|上采样方式|最后一步一次性32倍上采样（粗糙）|逐步2倍上采样，分层复原尺寸（精细）|
|细节保留能力|差，一步放大丢失大量浅层细节|优，跳跃连接拼接浅层细节特征|
|网络架构|单编码器\+单次上采样解码器|对称编码器\-解码器\+跳跃连接|
|小目标/边缘分割|效果较差，边缘模糊、小目标丢失|效果优异，边缘清晰、小目标完整|
|参数量|参数量少，结构简单|参数量中等，结构更完善|

🧠 核心升级逻辑：FCN 直接32倍放大特征图，空间细节严重丢失；U\-Net通过**逐步上采样\+跳跃连接**，将编码器浅层的高分辨率细节特征融入解码器，大幅提升分割精度，是医学影像、小目标分割的主流模型。

---

# 十、本日关键记忆点

|编号|知识点|核心总结|
|---|---|---|
|①|FCN 核心思想|去除全连接层、全卷积结构，实现任意尺寸输入、逐像素密集预测|
|②|FCN架构三件套|截断ResNet主干 → 1×1卷积分类投影 → 转置卷积上采样复原尺寸|
|③|1×1卷积作用|不改变尺寸，仅压缩通道数，将512维特征转为21类分类分数|
|④|转置卷积核心参数|64核\+stride32\+padding16，精准实现32倍上采样，匹配原图尺寸|
|⑤|双线性核初始化价值|规避随机初始化的混乱上采样，加速模型收敛，提升训练稳定性|
|⑥|双线性核特征|金字塔权重分布，中心权重最大，向四周线性衰减，实现平滑插值|
|⑦|逐像素损失函数|对所有像素计算交叉熵，取空间维度均值作为全局损失|
|⑧|预测可视化流程|前向推理→像素argmax分类→颜色映射→输出彩色分割图|
|⑨|32倍下采样来源|ResNet\-18包含5次stride=2下采样，总缩放倍数$2^5=32$|
|⑩|FCN与U\-Net核心差异|FCN单次粗上采样丢失细节；U\-Net分层上采样\+跳跃连接，分割精度更高|
