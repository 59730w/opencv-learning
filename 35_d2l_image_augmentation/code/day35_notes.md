# Day35：图片增广/数据增广

**核心主题**：数据增广的动机 → RandomHorizontalFlip/VerticalFlip → RandomResizedCrop → ColorJitter → Compose 组合 → CIFAR\-10 上对比有无增广的训练效果

---

## 一、为什么需要数据增广？

### 1\.1 三种防过拟合手段的关系

|手段|作用层面|机制|对应课程|
|---|---|---|---|
|**权重衰减（L2）**|损失函数|惩罚大参数 → 约束模型复杂度|Day24|
|**Dropout**|神经元|随机丢弃 → 打破共适应|Day24|
|**数据增广**|**数据本身**|生成更多"虚拟"训练样本 → 数据多了自然不过拟合|Day35 ← 今天|

🎯 数据增广是最根本的防过拟合手段——**让模型看到的"世界"比实际数据更丰富**。前两种是"约束模型"，这一种是"丰富数据"。

### 1\.2 核心思想

一张猫的照片，左右翻转后**仍然是一只猫**——语义不变，但像素值全变了。利用这种"语义不变性"，通过对训练图片施加随机变换，凭空创造出大量新的训练样本。

原始训练集：1 张猫照片

增广后（在线生成）：水平翻转版 \+ 垂直翻转版 \+ 裁剪版 \+ 颜色偏移版 \+ 组合版

→ 模型每个 epoch 看到的都是略微不同的图片 → 永远不会看到完全相同的样本两次

---

## 二、加载原始图片

```python
d2l.set_figsize()
img = d2l.Image.open(r"D:\DL_code\d2l-zh\pytorch\img\cat1.jpg")
d2l.plt.imshow(img)

```

使用 `PIL.Image.open` 加载图片作为后续所有增广操作的"原材料"。

## 三、增广可视化工具：apply\(\) 函数

```python
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    """对同一张图重复应用增广 aug，展示 num_rows×num_cols 个结果"""
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

```

|参数|含义|本实验取值|
|---|---|---|
|img|原始 PIL 图片|猫图|
|aug|增广变换（随机操作）|见下文各节|
|num\_rows × num\_cols|展示网格|2×4 = 8 个随机结果|
|scale|图片缩放比例|1\.5|

🔑 关键理解：每次 `aug(img)` 都会随机产生不同结果——8 次调用，8 个不同的输出，直观感受增广的随机性。

## 四、翻转增广

### 4\.1 水平翻转 RandomHorizontalFlip\(\)

```python
apply(img, torchvision.transforms.RandomHorizontalFlip())

```

|参数|默认值|含义|
|---|---|---|
|p|0\.5|以 50% 概率水平翻转|

效果：猫脸朝左 → 猫脸朝右（镜像）。猫还是那只猫，语义完全不变。

### 4\.2 垂直翻转 RandomVerticalFlip\(\)

```python
apply(img, torchvision.transforms.RandomVerticalFlip())

```

|参数|默认值|含义|
|---|---|---|
|p|0\.5|以 50% 概率上下翻转|

⚠️ 垂直翻转不一定保持语义——上下颠倒的猫不常见。对森林树种分类来说，树叶上下翻转也有意义（拍摄角度不同）；但对通用场景（如 ImageNet），垂直翻转通常不用。

## 五、随机裁剪/缩放 RandomResizedCrop

```python
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200),           # 最终输出尺寸
    scale=(0.1, 1),       # 裁剪面积占原图的比例范围
    ratio=(0.5, 2)        # 裁剪宽高比的范围
)
apply(img, shape_aug)

```

|参数|取值|含义|
|---|---|---|
|size|(200, 200\)|最终统一输出 200×200|
|scale|(0\.1, 1\)|随机裁原图 10%\~100% 面积的区域|
|ratio|(0\.5, 2\)|裁剪区域的宽/高在 0\.5\~2 之间随机选|

**操作流程**

```text
原图 (任意尺寸)
  │
  ├─ ① 随机选裁剪区域的面积比例（如 0.6 → 裁 60% 面积）
  ├─ ② 随机选裁剪区域的宽高比（如 1.3 → 宽是高的 1.3 倍）
  ├─ ③ 在图上随机选位置裁剪
  └─ ④ resize 到 (200, 200)

```

🎯 这是最强大的空间增广——模型被迫学会识别"近景的一小片树叶"和"远景的一整棵树"，尺度和构图的鲁棒性大幅提升。

## 六、颜色增广 ColorJitter

### 6\.1 亮度（Brightness）

```python
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))

```

|参数|取值|含义|
|---|---|---|
|brightness|0\.5|亮度在 [1\-0\.5, 1\+0\.5\] = [0\.5, 1\.5\] 之间随机变化|

亮度 0\.5 → 模拟阴天/阳光直射。森林场景中光照条件千变万化，这是最重要的颜色增广。

### 6\.2 色相（Hue）

```python
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))

```

|参数|取值|含义|
|---|---|---|
|hue|0\.5|色相在 [\-0\.5, 0\.5\] 之间随机偏移|

色相偏移 0\.5 → 颜色可能偏绿/偏紫。不同季节树叶颜色不同（春绿秋红），森林场景中非常有价值。

### 6\.3 四合一组合

```python
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

```

|参数|全部取值 0\.5|效果|
|---|---|---|
|brightness|0\.5|亮度在 [0\.5, 1\.5\] 浮动|
|contrast|0\.5|对比度在 [0\.5, 1\.5\] 浮动|
|saturation|0\.5|饱和度在 [0\.5, 1\.5\] 浮动|
|hue|0\.5|色相在 [\-0\.5, 0\.5\] 偏移|

## 七、组合增广 Compose

```python
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),  # ① 先随机翻转
    color_aug,                                      # ② 再调颜色
    shape_aug                                       # ③ 最后随机裁剪
])
apply(img, augs)

```

|执行顺序|操作|说明|
|---|---|---|
|①|RandomHorizontalFlip|空间变换先行|
|②|ColorJitter|颜色变换|
|③|RandomResizedCrop|裁剪 \+ 缩放|

📌 Compose 的顺序很重要：先做翻转（不影响颜色），再做颜色变换（不影响空间），最后裁剪到统一尺寸。三合一的效果远比单一增广丰富。

## 八、训练集 vs 测试集：增广策略的铁律

```python
# 训练集：有增广
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
# 测试集：只有 ToTensor（不做任何增广！）
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

```

🔑 铁律：只对训练集做增广，测试集不做！

测试集代表"真实世界中模型将要面对的未见过的图片"——真实世界不会帮你做翻转/颜色抖动。测试时保持数据原样，才能正确评估泛化能力。

## 九、CIFAR\-10 增广训练实战

### 9\.1 CIFAR\-10 数据集

```python
all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

```

|属性|值|
|---|---|
|图片尺寸|32×32 彩色（3 通道）|
|训练集|50,000 张|
|测试集|10,000 张|
|类别|10 类（飞机、汽车、鸟、猫、鹿、狗、蛙、马、船、卡车）|

CIFAR\-10 的图片是 PIL Image 格式，可以直接送入 transforms\.Compose 的 Pipeline。

### 9\.2 数据加载函数

```python
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(
        root="../data", train=is_train,
        transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
    return dataloader
```

|参数|训练集|测试集|
|---|---|---|
|is\_train|True|False|
|augs|RandomHorizontalFlip \+ ToTensor|仅 ToTensor|
|shuffle|True|False|

### 9\.3 多 GPU 训练函数 train\_ch13

本课首次使用了 `nn.DataParallel` 做多 GPU 训练：

```python
net = nn.DataParallel(net, device_ids=devices).to(devices[0])

```

|对比|之前的 train\_ch6|本课的 train\_ch13|
|---|---|---|
|设备|单个 GPU (d2l\.try\_gpu\(\)\)|多 GPU (d2l\.try\_all\_gpus\(\)\)|
|模型包装|无|nn\.DataParallel(net\)|
|损失处理|l\.backward\(\)|l\.sum\(\)\.backward\(\)|

📌 多 GPU 训练的核心就是把模型用 `nn.DataParallel` 包一层。对于本机单张 RTX3050 的环境，devices 列表中只有一个设备，DataParallel 退化为单 GPU，不会影响训练逻辑。

### 9\.4 模型：ResNet\-18 适配 CIFAR\-10

```python
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
# d2l.resnet18(num_classes=10, in_channels=3)

```

|参数|值|说明|
|---|---|---|
|num\_classes|10|CIFAR\-10 的 10 类|
|in\_channels|3|RGB 三通道|

### 9\.5 训练执行

```python
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter  = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
train_with_data_aug(train_augs, test_augs, net)

```

|超参数|取值|说明|
|---|---|---|
|batch\_size|256|—|
|lr|0\.001|Adam 优化器，小学习率|
|num\_epochs|10|CIFAR\-10 50K 样本，10 轮足够看到趋势|

## 十、有增广 vs 无增广的效果对比（概念层面）

|对比维度|无增广|有增广|
|---|---|---|
|每个 epoch 看到的样本|固定 50,000 张|每次都不一样（在线随机变换）|
|过拟合发生时间|早（几个 epoch 后 test acc 不再上升）|晚（test acc 持续上升更多 epoch）|
|最终 test accuracy|较低|更高|
|训练速度|快|稍慢（CPU 做增广变换）|
|模型鲁棒性|弱（对翻转/光照敏感）|强（见过各种变体）|

🎯 数据增广本质上是在"免费"增加数据集大小——不需要采集和标注新数据，仅靠现有的 50,000 张图片就能产生近乎无限的变体。

## 十一、所有增广技术速查表

|变换|PyTorch API|关键参数|适用场景|
|---|---|---|---|
|水平翻转|RandomHorizontalFlip(p=0\.5\)|p|通用（猫/狗/树等）|
|垂直翻转|RandomVerticalFlip(p=0\.5\)|p|慎用（树叶/航拍可用）|
|随机裁剪缩放|RandomResizedCrop(size, scale, ratio\)|scale=(0\.1,1\)|✅ 必加|
|亮度|ColorJitter(brightness=0\.5\)|brightness|光照变化大的场景|
|对比度|ColorJitter(contrast=0\.5\)|contrast|雾天/阴天|
|饱和度|ColorJitter(saturation=0\.5\)|saturation|季节变化|
|色相|ColorJitter(hue=0\.5\)|hue ∈ [0,0\.5\]|色调偏移|
|组合|Compose([\.\.\.\]\)|顺序重要|✅ 用于训练集|
|转张量|ToTensor\(\)|—|训练\+测试都要|

## 十二、为森林 Demo 定制的理想训练 Pipeline

结合你的森林树种分类场景，推荐训练集增广配置：

```python
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),      # 模拟近景远景
    transforms.RandomHorizontalFlip(),                         # 树叶从左边/右边拍
    transforms.RandomVerticalFlip(),                           # 仰拍/俯拍
    transforms.ColorJitter(
        brightness=0.4,      # 光照：清晨/正午/阴天
        contrast=0.4,        # 对比度：雾天
        saturation=0.4,      # 饱和度：季节变化
        hue=0.1              # 色调：不同树种偏色（hue 不宜太大）
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],          # ImageNet 统计值
                         std=[0.229, 0.224, 0.225])           # （配合预训练模型）
])
test_augs = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

```

|细节|说明|
|---|---|
|scale=(0\.5, 1\.0\)|森林场景裁剪比例不宜太极端（0\.1 会裁到只有叶脉纹理，反而增加学习难度）|
|hue=0\.1|树种分类中颜色是重要特征，色相偏移太大可能破坏树种区分信息|
|Normalize|使用 ImageNet 的均值和标准差——配合预训练 ResNet 做微调|

## 十三、与之前课程的联系

|之前课程|在本课的体现|
|---|---|
|Day23 过拟合|今天用数据增广正面解决过拟合——增加"等效数据量"|
|Day24 权重衰减 \+ Dropout|三种防过拟合手段齐备：L2 约束参数 / Dropout 打破共适应 / 增广丰富数据|
|Day32 AlexNet|CIFAR\-10 曾经是 AlexNet 的首选基准数据集|
|Day34 ResNet|d2l\.resnet18(10, 3\)——昨天刚学的 ResNet 今天就用上了|
|Day21 Softmax|CrossEntropyLoss 和训练循环结构完全一样|

## 十四、本日关键记忆点

|编号|知识点|一句话|
|---|---|---|
|①|数据增广的本质|利用语义不变性在线生成虚拟样本 → 等效扩大数据集|
|②|铁律|训练集增广，测试集不做——测试集保持"真实世界"样貌|
|③|RandomResizedCrop|最重要的空间增广——同时改变尺度、构图和位置|
|④|ColorJitter 四参数|brightness/contrast/saturation/hue——四个维度独立控制|
|⑤|Compose 顺序|空间变换 → 颜色变换 → ToTensor（先空间后颜色，先 PIL 后 Tensor）|
|⑥|增广效果|有增广 → 过拟合更晚 → test accuracy 更高|
|⑦|森林 Demo 的 hue|只设 0\.1 而非 0\.5——颜色是树种识别的关键特征，不宜过度扭曲|
|⑧|ResNet \+ CIFAR\-10|本课首次把 ResNet\-18 和经典数据集结合训练|
