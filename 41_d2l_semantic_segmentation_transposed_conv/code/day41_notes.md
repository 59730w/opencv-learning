# Day41：语义分割与数据集 \+ 转置卷积

**核心主题**：语义分割任务定义 → Pascal VOC 2012 数据集 → RGB 色图 → 类别索引 → 随机裁剪 → 自定义分割数据集 → 转置卷积原理与实现

---

# 第一部分：语义分割

## 1\.1 图像任务的「粒度」演进

|任务|输出|粒度|典型方法|
|---|---|---|---|
|图像分类|1 个类别标签|整张图|ResNet|
|目标检测|类别 \+ 边界框|多个目标框|SSD / YOLO（Day40）|
|**语义分割**|**每个像素一个类别**|**逐像素**|**FCN / U\-Net（后续）**|

图像分类：     [一张图\] → "这是一只猫"（1个标签）

目标检测：     [一张图\] → "猫在这里"（框\+标签）

语义分割：     [一张图\] → 每个像素标上"猫/背景/桌子"（像素级标签图）

🎯 语义分割回答的问题是：**这张图里每个像素分别属于哪个类别**。这在森林监测中对应**植被覆盖度分割、林区土地覆盖分类**，和林业遥感研究方向高度契合。

## 1\.2 语义分割 vs 实例分割

|类型|区分不同实例？|示例|
|---|---|---|
|**语义分割**|❌ 不区分（所有"猫"算同一类）|所有猫像素都标为"猫"|
|**实例分割**|✅ 区分（猫1、猫2 分开）|每只猫单独一个掩码|

语义分割只关心「像素属于哪类」，不管同一类里有几个物体。Mask R\-CNN（Day39 学的）属于实例分割。

## 1\.3 Pascal VOC 2012 数据集

### 基本信息

|属性|值|
|---|---|
|名称|VOC2012（Visual Object Classes）|
|图像类型|自然图像（含人、动物、车辆、室内物体等）|
|标注类型|像素级分割标签（PNG 色图）|
|类别数|21 类（background \+ 20 个物体类别）|
|训练/验证|`train.txt` / `val.txt` 列出图片名|

### 21 个类别

```python
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

```

索引：0 → 1 → 2 → \.\.\. → 20

类别：background → aeroplane → bicycle → \.\.\. → tv/monitor

### 数据目录结构

```Plain Text
VOCdevkit/VOC2012/
├── JPEGImages/          ← 原始图像（.jpg）
├── SegmentationClass/   ← 像素级分割标签（.png 色图）
└── ImageSets/
    └── Segmentation/
        ├── train.txt    ← 训练图像名列表
        └── val.txt      ← 验证图像名列表

```

## 1\.4 读取图像和标签（read\_voc\_images）

```python
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()          # 图片名列表
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))      # 原图
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png'), mode))  # 分割标签
    return features, labels
```

|读取对象|文件路径|格式|
|---|---|---|
|原图|JPEGImages/\{name\}\.jpg|RGB 彩色图|
|标签|SegmentationClass/\{name\}\.png|RGB 色图（每个像素颜色对应一个类别）|

分割标签是 PNG 色图：每个像素的颜色通过 VOC\_COLORMAP 映射到类别索引。

## 1\.5 色图 → 类别索引的转换（核心）

### 颜色映射表

```python
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

```

21 种颜色，每种对应一个类别。比如 [0,0,0\]（黑色）= 背景，[128,0,0\]（暗红）= 飞机。

### 构建颜色 → 索引查找表

```python
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)   # 256³ 个可能颜色
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label
```

**RGB 编码公式**：$\text{index} = (R \times 256 + G) \times 256 + B$

示例：[128, 0, 0\] → (128×256\+0\)×256\+0 = 8388608

|关键点|说明|
|---|---|
|表大小 256³|覆盖所有可能的 RGB 组合（256×256×256）|
|仅21个有效位置|只有 VOC 的 21 种颜色对应有效类别，其余默认为0|
|O(1\) 查找|直接索引访问，远快于逐像素颜色比对|

### 标签色图转类别索引图

```python
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')  # (C,H,W) → (H,W,C)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])                                    # (H,W) 编码索引
    return colormap2label[idx]                                     # (H,W) 类别索引图

```

**转换流程**：标签色图 (3,H,W\) → 维度变换 (H,W,C\) → 整数编码 → 查表 → (H,W\) 类别索引图

**核心意义**：模型训练输出逐像素类别概率，标签必须是二维类别索引图，而非三维RGB色图，这一步是分割训练的前置核心操作。

## 1\.6 随机裁剪（voc\_rand\_crop）——原图标签同步裁剪

```python
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))          # ① 生成随机裁剪参数（基于原图）
    feature = torchvision.transforms.functional.crop(feature, *rect)  # ② 裁原图
    label = torchvision.transforms.functional.crop(label, *rect)      # ③ 裁标签（同位置）
    return feature, label

```

关键逻辑：用同一个裁剪矩形 rect 同步裁剪原图和标签，严格保证像素位置一一对应，杜绝错位问题。

## 1\.7 自定义分割数据集 VOCSegDataset

```python
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir):
        # ① 预处理：ImageNet 标准化
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        # ② 过滤尺寸不足的图片
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)   # 像素归一化 [0,1] + 标准化

    def filter(self, imgs):
        # 只保留尺寸 ≥ 裁剪尺寸的图片
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)   # 同步随机裁剪
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

```

|组件|作用|
|---|---|
|filter|过滤小于裁剪尺寸的图片，避免裁剪报错|
|normalize\_image|原图归一化至[0,1\]，再用ImageNet均值方差标准化|
|voc\_rand\_crop|原图、标签同步随机裁剪，数据增广|
|voc\_label\_indices|RGB标签色图转为二维类别索引图|

**返回张量格式**

feature: `(3, crop_h, crop_w)` 标准化原图

label: `(crop_h, crop_w)` 类别索引图（0\~20整型索引）

## 1\.8 数据加载验证

```python
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)    # 训练集
voc_test = VOCSegDataset(False, crop_size, voc_dir)    # 验证集
batch_size = 8
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                         drop_last=True, num_workers=0)
for X, Y in train_iter:
    print(X.shape)   # torch.Size([8, 3, 320, 480])
    print(Y.shape)   # torch.Size([8, 320, 480])
    break
```

|张量|形状|含义|
|---|---|---|
|X|(8, 3, 320, 480\)|8张320×480 RGB批量原图|
|Y|(8, 320, 480\)|8张逐像素类别索引标签图|

`drop_last=True`：丢弃最后不完整批次，避免BN层batch\_size=1出现训练报错。

---

# 第二部分：转置卷积

## 2\.1 为什么需要转置卷积？

语义分割要求输出特征图与输入原图尺寸完全一致。普通卷积、池化会持续下采样缩小特征图，因此需要**可学习的上采样**恢复空间分辨率。

下采样（普通卷积/池化）：大图 → 小图（提取抽象特征）

上采样（转置卷积）：小图 → 大图（恢复空间细节）

🔑 核心定义：转置卷积是普通卷积的**形状逆操作**，可将下采样缩小的特征图恢复为原图尺寸。

## 2\.2 转置卷积计算原理

转置卷积并非数学意义上的卷积逆运算，而是**输入元素扩散累加**的特殊卷积操作：每个输入元素乘以卷积核，扩散到输出对应区域并累加。

```python
def trans_conv(X, K):
    """转置卷积朴素实现：输入元素扩散累加"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

```

### 手算示例

输入 X(2×2\)、卷积核 K(2×2\)

输出尺寸：$2+2-1=3$ → 3×3

核心直觉：普通卷积是「多输入加权求和得单输出」，转置卷积是「单输入扩散贡献多输出」。

## 2\.3 nn\.ConvTranspose2d 官方实现验证

```python
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))   # 结果与手写朴素实现完全一致

```

## 2\.4 转置卷积 Padding 与 Stride \& 尺寸公式

**转置卷积输出尺寸公式**

$H_{out} = (H_{in} - 1) \times stride - 2 \times padding + kernel\_size$

$W_{out} = (W_{in} - 1) \times stride - 2 \times padding + kernel\_size$

|配置|代入公式|输出尺寸|
|---|---|---|
|无padding，stride=1|(2\-1\)×1\-0\+2=3|3×3|
|padding=1，stride=1|(2\-1\)×1\-2\+2=1|1×1|
|stride=2|(2\-1\)×2\-0\+2=4|4×4|

stride=2 实现特征图放大上采样，输入元素间插入零值空隙后扩散卷积。

### 卷积与转置卷积形状互逆验证

```python
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)   # 下采样
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)  # 上采样
print(tconv(conv(X)).shape == X.shape)   # True，尺寸完全复原

```

这是转置卷积在分割网络中的核心价值：完美复原卷积下采样丢失的空间尺寸。

## 2\.5 转置卷积 = 卷积权重矩阵的转置

卷积运算可等价为矩阵乘法，转置卷积本质是卷积权重矩阵的转置运算。

普通卷积：$Y_{vec} = W \cdot X_{vec}$ （高维 → 低维，下采样）

转置卷积：$Z_{vec} = W^T \cdot Y_{vec}$ （低维 → 高维，上采样）

「转置卷积」名称的核心由来：复用同一卷积核，仅对权重矩阵做转置运算，实现尺寸逆变换。

## 2\.6 各类上采样方法对比

|方法|可学习参数|特点说明|
|---|---|---|
|最近邻插值|❌ 无|像素复制放大，锯齿感严重，效果差|
|双线性插值|❌ 无|固定平滑放大，无法适配任务特性|
|转置卷积|✅ 有|网络自适应学习上采样规则，适配分割任务|
|反卷积|—|俗称，本质等同于转置卷积|

---

# 第三部分：与之前课程的联系

|往期课程|本节课对应体现|
|---|---|
|Day29 卷积基础|转置卷积是普通卷积的矩阵转置形式，同源同核|
|Day30 Padding/Stride|转置卷积同款参数，作用效果与普通卷积完全相反|
|Day31 池化下采样|池化缩小尺寸，转置卷积放大尺寸，上下采样互补|
|Day34 BN/归一化|VOC数据集复用ImageNet标准化统计值|
|Day35 数据增广|分割专属随机裁剪，必须同步操作原图与标签|
|Day38 目标检测|检测框级粒度 → 分割像素级粒度，任务精度升级|
|Day40 SSD检测|SSD多尺度下采样，分割需转置卷积上采样复原尺寸|

# 第四部分：语义分割完整 Pipeline

Pascal VOC 2012 数据集

　│

　├── JPEGImages（原图 RGB）

　└── SegmentationClass（标签色图 RGB，像素颜色对应类别）

　　　　│

　　　　▼

　　色图 → 类别索引（colormap2label 查表转换）

　　　　│

　　　　▼

　　随机同步裁剪（原图\+标签同位置裁剪）

　　　　│

　　　　▼

　　原图归一化 \+ 标签转为索引图

　　　　│

　　　　▼

　VOCSegDataset → DataLoader → \(原图X, 索引标签Y\)

　　　　│

　　　　▼

FCN / U\-Net：卷积下采样提取特征 → 转置卷积上采样复原尺寸 → 逐像素分类

# 第五部分：本日关键记忆点

## 语义分割与数据集

|编号|知识点|核心总结|
|---|---|---|
|①|语义分割定义|逐像素分类，为图像每一个像素分配对应类别|
|②|语义/实例分割区别|语义不区分同类个体，实例分割区分不同物体实例|
|③|VOC标签格式|PNG彩色标签图，21种固定颜色对应21个类别|
|④|色图转索引原理|RGB三通道编码为整数，通过256³查表快速映射类别|
|⑤|索引表设计优势|全覆盖RGB色域，仅21个有效类别，O(1\)极速查找|
|⑥|同步裁剪机制|同一裁剪参数同步处理原图和标签，保证像素对齐|
|⑦|图片过滤规则|仅保留尺寸大于等于裁剪尺寸的图像，防止训练报错|
|⑧|数据张量形状|输入X(B,3,H,W\)，标签Y(B,H,W\)二维索引图|

## 转置卷积

|编号|知识点|核心总结|
|---|---|---|
|⑨|转置卷积作用|可学习的特征上采样，复原卷积下采样丢失的空间尺寸|
|⑩|计算核心原理|输入每个元素乘卷积核，扩散累加至输出对应区域|
|⑪|输出尺寸公式|$H_{out}=(H-1)\times s-2p+k$|
|⑫|名称由来|本质是普通卷积权重矩阵的转置矩阵运算|
|⑬|卷积互逆特性|同参数Conv2d\+ConvTranspose2d可完全复原特征图尺寸|
|⑭|上采样方法对比|插值法固定不可学习，转置卷积可自适应学习，适配分割任务|
