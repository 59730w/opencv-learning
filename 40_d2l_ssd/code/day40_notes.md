# Day40：单发多框检测 SSD

**核心主题**：SSD 单阶段检测器完整实现——多尺度特征图 → 多尺度锚框 → 类别\+偏移量同时预测 → 多任务损失 → NMS 后处理

---

# 一、SSD 是什么？

## 1\.1 核心思想

**SSD（Single Shot MultiBox Detector，单发多框检测）**：用一次前向传播，直接在**多个尺度的特征图**上预测目标的**类别**和**边界框偏移量**。

|对比维度|两阶段（R\-CNN 家族，Day39）|单阶段（SSD/YOLO）|
|---|---|---|
|流程|先提提议区域 → 再分类精修|**一步直接预测**|
|速度|慢|**快**（一次前向）|
|精度|较高|略低（但够用）|
|代表|Faster R\-CNN|**SSD、YOLO**|

## 1\.2 SSD 如何实现「多尺度」？

回顾 Day39 学的多尺度锚框思想：**不同层的特征图感受野不同 → 用不同特征图检测不同大小目标**。

SSD 把这个思想落地为：**5 个不同尺度的特征图 → 5 组不同大小的锚框 → 5 个检测头分别预测**。

输入图像

│

▼

基础网络 \(base\_net\) ──▶ 特征图1 \(尺度1\) ──▶ 检测头1（小锚框→小目标）

│

▼

下采样块1 ──▶ 特征图2 \(尺度2\) ──▶ 检测头2

│

▼

下采样块2 ──▶ 特征图3 \(尺度3\) ──▶ 检测头3

│

▼

下采样块3 ──▶ 特征图4 \(尺度4\) ──▶ 检测头4

│

▼

自适应池化 ──▶ 特征图5 \(尺度5\) ──▶ 检测头5（大锚框→大目标）

---

# 二、SSD 的五大基础组件

## 2\.1 类别预测器 cls\_predictor

```python
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

```

|参数|含义|
|---|---|
|num\_inputs|输入通道数（特征图通道）|
|num\_anchors|每个像素位置的锚框数|
|num\_classes|目标类别数|
|输出通道|num\_anchors × (num\_classes \+ 1\)|

**为什么是 num\_anchors × (num\_classes \+ 1\)？**

每个锚框要预测 类别数 \+ 1（加 1 是背景类）个概率。比如 5 个锚框 × 10 类 = 5×(10\+1\) = 55 个输出通道。每个通道对应「某个锚框属于某个类别的分数」。

## 2\.2 边界框预测器 bbox\_predictor

```python
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

```

每个锚框预测 4 个偏移量（Δx, Δy, Δw, Δh，Day38 学的偏移量编码）→ 输出通道 = num\_anchors × 4。

## 2\.3 展平与拼接 flatten\_pred \+ concat\_preds

```python
def flatten_pred(pred):
    # (batch, C, H, W) → (batch, H*W*C)
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    # 把多个尺度特征图的预测拼接到一起
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

```

|函数操作|作用|
|---|---|
|permute(0,2,3,1\)|通道维移到最后：(B,C,H,W\) → (B,H,W,C\)|
|flatten(start\_dim=1\)|除 batch 外全部展平：(B,H,W,C\) → (B, H×W×C\)|
|concat(preds, dim=1\)|5 个尺度的预测沿特征维拼接|

**为什么用 permute 再 flatten？** 保证「同一位置的锚框预测」在展平后相邻排列，方便后续 reshape 成 (batch, num\_anchors, num\_classes\+1\)。

## 2\.4 下采样块 down\_sample\_blk

```python
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))   # Day34 学的 BN
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))                    # 空间减半
    return nn.Sequential(*blk)

```

**结构**：2×(Conv3×3\+BN\+ReLU\) \+ MaxPool2d\(2\)

输入 (B, in\_ch, H, W\) → [Conv→BN→ReLU\]×2 → (B, out\_ch, H, W\) → MaxPool → (B, out\_ch, H/2, W/2\)

每个块空间减半（MaxPool stride=2），通道数翻倍——和 VGG/ResNet 的「空间减半、通道翻倍」哲学一致。

## 2\.5 基础网络 base\_net

```python
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]   # 3→16→32→64
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

```

输入 (B, 3, 256, 256\)

→ down\_sample\_blk(3, 16\):    (B, 16, 128, 128\)

→ down\_sample\_blk(16, 32\):   (B, 32, 64, 64\)

→ down\_sample\_blk(32, 64\):   (B, 64, 32, 32\)

base\_net 是 SSD 的主干网络（backbone），负责提取基础特征。真实 SSD 用 VGG 作 backbone，这里用简化版。

# 三、5 个尺度的块配置

## 3\.1 get\_blk——根据索引返回不同的块

```python
def get_blk(i):
    if i == 0:
        blk = base_net()              # 尺度1：基础网络（32×32 特征图）
    elif i == 1:
        blk = down_sample_blk(64, 128)  # 尺度2：16×16
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))  # 尺度5：1×1
    else:
        blk = down_sample_blk(128, 128)   # 尺度3、4：8×8、4×4
    return blk

```

## 3\.2 5 个尺度一览

|尺度 i|块|输入通道|特征图尺寸|锚框大小 sizes|
|---|---|---|---|---|
|0|base\_net|64|32×32|[0\.2, 0\.272\]|
|1|down(64→128\)|128|16×16|[0\.37, 0\.447\]|
|2|down(128→128\)|128|8×8|[0\.54, 0\.619\]|
|3|down(128→128\)|128|4×4|[0\.71, 0\.79\]|
|4|AdaptiveMaxPool(1,1\)|128|1×1|[0\.88, 0\.961\]|

## 3\.3 锚框参数

```python
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1   # 2 + 3 - 1 = 4

```

每个像素 4 个锚框（Day38 学的公式：num\_sizes \+ num\_ratios \- 1）：

sizes[0\]=0\.2 × ratios[0\]=1   → 小正方形

sizes[1\]=0\.272 × ratios[0\]=1 → 大正方形

sizes[0\]=0\.2 × ratios[1\]=2   → 宽矩形

sizes[0\]=0\.2 × ratios[2\]=0\.5 → 窄矩形

📐 规律：sizes 随尺度增大而增大（0\.2 → 0\.88），即深层特征图用更大锚框检测更大目标——正是 Day39 多尺度思想。

# 四、blk\_forward——单尺度块的前向

```python
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)                              # ① 块前向 → 特征图
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)  # ② 生成锚框
    cls_preds = cls_predictor(Y)            # ③ 类别预测
    bbox_preds = bbox_predictor(Y)          # ④ 偏移量预测
    return (Y, anchors, cls_preds, bbox_preds)

```

每个尺度块返回 4 样东西：

|返回值|内容|
|---|---|
|Y|该尺度特征图（作为下一块的输入）|
|anchors|该尺度生成的锚框（multibox\_prior，Day38 学的）|
|cls\_preds|类别预测（形状 (B, num\_anchors×(C\+1\), H, W\)）|
|bbox\_preds|偏移量预测（形状 (B, num\_anchors×4, H, W\)）|

# 五、TinySSD 主网络

## 5\.1 初始化

```python
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]   # 5个尺度块输入通道
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

```

用 setattr(self, f'blk\_\{i\}', \.\.\.\) 动态注册 5 个块 \+ 5 个类别头 \+ 5 个框头——Day27 学的动态注册子模块技巧。

## 5\.2 前向传播

```python
def forward(self, X):
    anchors, cls_preds, bbox_preds = [None]*5, [None]*5, [None]*5
    for i in range(5):
        X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
            X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
            getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
    anchors = torch.cat(anchors, dim=1)              # 拼接所有尺度锚框
    cls_preds = concat_preds(cls_preds)              # (B, 总锚框数×(C+1))
    cls_preds = cls_preds.reshape(
        cls_preds.shape[0], -1, self.num_classes + 1)  # (B, 总锚框数, C+1)
    bbox_preds = concat_preds(bbox_preds)            # (B, 总锚框数×4)
    return anchors, cls_preds, bbox_preds

```

## 5\.3 形状流转验证（核心）

```python
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)
print('output anchors:', anchors.shape)     # torch.Size([1, 5444, 4])
print('output class preds:', cls_preds.shape)  # torch.Size([32, 5444, 2])
print('output bbox preds:', bbox_preds.shape)  # torch.Size([32, 21776])

```

**5444 个锚框怎么算出来的？**

|尺度|特征图|每像素锚框|锚框数|
|---|---|---|---|
|0|32×32|4|32×32×4 = 4096|
|1|16×16|4|16×16×4 = 1024|
|2|8×8|4|8×8×4 = 256|
|3|4×4|4|4×4×4 = 64|
|4|1×1|4|1×1×4 = 4|
|**合计**|||**5444**|

类别预测：5444 × (1\+1\) = 10888 → reshape 后 (32, 5444, 2\)

框预测：  5444 × 4 = 21776

🎯 一次前向传播同时输出 5444 个锚框的类别概率和偏移量——这就是「单发（Single Shot）」的含义。

# 六、训练

## 6\.1 多任务损失 calc\_loss

```python
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # ① 类别损失：交叉熵（Day38 multibox_target 生成的类别标签）
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # ② 边界框损失：L1 损失（只算掩码为1的正样本锚框）
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

```

|损失|损失函数|输入来源|
|---|---|---|
|类别损失|nn\.CrossEntropyLoss|cls\_preds（网络预测）\+ cls\_labels（锚框类别标签）|
|框损失|nn\.L1Loss（MAE）|bbox\_preds（网络预测）\+ bbox\_labels（偏移量标签）|
|掩码 bbox\_masks|乘上掩码|只对正样本锚框计算框损失，背景锚框的偏移量不计|

**总损失 = 类别交叉熵损失 \+ 边界框 L1 损失（掩码过滤）**

训练循环中，锚框标签由 Day38 学的 d2l\.multibox\_target\(anchors, Y\_train\) 生成：

```python
bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y_train)

```

## 6\.2 评估指标

```python
def cls_eval(cls_preds, cls_labels):
    # 类别错误数：预测类别 ≠ 真实类别
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 框误差：|真实-预测|×掩码 的绝对值之和
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

```

|指标|含义|
|---|---|
|class error|1 \- 准确率（预测错误的锚框占比）|
|bbox mae|平均绝对误差（掩码过滤后）|

## 6\.3 训练循环

```python
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)   # Day38 学的香蕉数据集
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

for epoch in range(20):
    net.train()
    for features, target in train_iter:
        X_train, Y_train = features.to(device), target.to(device)
        anchors, cls_preds, bbox_preds = net(X_train)          # ① 前向
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y_train)  # ② 生成锚框标签
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)   # ③ 多任务损失
        l.mean().backward()                                    # ④ 反向
        trainer.step()                                         # ⑤ 更新
        metric.add(cls_eval(...), cls_labels.numel(), bbox_eval(...), bbox_labels.numel())

```

训练循环和之前分类任务几乎一样，只是多了一步 multibox\_target（生成锚框的类别和偏移量标签）和组合损失。

# 七、预测推理（NMS 后处理）

```python
def predict(X, net, device):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))      # ① 前向
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)  # ② 类别概率
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)  # ③ NMS 去重
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]     # ④ 过滤背景
    return output[0, idx]

```

网络输出 → softmax 类别概率 → multibox\_detection（Day38 学的：偏移量解码 \+ NMS 抑制重复框 \+ 低置信度过滤）→ 最终检测结果

display 函数按阈值（0\.9）画框并标注置信度：

```python
def display(img, output, threshold):
    ...
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]  # 归一化→像素
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

```

🎯 预测流程完全复用了 Day38 学的 multibox\_detection（NMS）——这就是为什么 Day38 那么重要，它是一整套检测后处理基础设施。

# 八、SSD 完整流程总结

**训练：**

输入图 → TinySSD 前向 → 5444 锚框 × (类别概率 \+ 偏移量\)

↓

multibox\_target\(锚框, 真实框\) → 锚框类别标签 \+ 偏移量标签

↓

损失 = CE\(类别\) \+ L1\(偏移量×掩码\) → 反向 → 更新

**预测：**

输入图 → TinySSD 前向 → softmax 概率 → multibox\_detection

（偏移量解码 → NMS 去重 → 过滤背景/低置信度）→ 最终框

# 九、课后练习：smooth L1 和 Focal Loss

这两个是检测/分类领域的重要损失函数，练习中画图理解。

## 9\.1 Smooth L1 损失

```python
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):     # |x| < 1/σ² 时用二次函数
            out.append(((scalar * i) ** 2) / 2)
        else:                              # |x| ≥ 1/σ² 时用线性函数
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

```

**公式定义**：

$\text{smooth\_l1}(x)=
\begin{cases}
\frac{(\sigma x)^2}{2}, & |x| < \frac{1}{\sigma^2} \\
|x| - \frac{1}{2\sigma^2}, & |x| \ge \frac{1}{\sigma^2}
\end{cases}$

|区间|函数形式|特点|
|---|---|---|
|小误差 ($\|x\| < 1/\sigma^2$)|二次函数|梯度平滑，训练稳定|
|大误差 ($\|x\| \ge 1/\sigma^2$)|线性函数|梯度恒定，对离群值鲁棒|

**对比**：

L1 (MAE\)：对大误差梯度恒为 1，鲁棒但 x=0 不可导

L2 (MSE\)：光滑但大误差梯度爆炸

Smooth L1：结合两者优点——小误差光滑，大误差鲁棒

实验：分别画 σ=10、1、0\.5 的三条曲线。σ 越大，二次区间越小（越接近 L1）；σ 越小，二次区间越大（越接近 L2）。实际 SSD 的 bbox 损失常用 Smooth L1。

## 9\.2 Focal Loss（聚焦损失）

```python
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

```

**公式**：$\text{Focal Loss} = -(1-p)^\gamma \cdot \log(p)$

标准交叉熵 = $-\log(p)$

Focal 在 CE 前乘了调制因子 $(1-p)^\gamma$

|γ (gamma\)|效果|
|---|---|
|0|退化为标准交叉熵|
|1|中等压制易分样本|
|5|强烈压制易分样本，聚焦难分样本|

**为什么用 Focal Loss？** 目标检测中大多数锚框是背景（负样本），正样本极少——类别极端不平衡。Focal Loss 让模型聚焦难分的正样本，抑制大量简单负样本的梯度贡献，解决类别不平衡。

**公式直觉**：

若 p 接近 1（易分样本）→ $(1-p)^\gamma$ 接近 0 → 损失被压制

若 p 接近 0（难分样本）→ $(1-p)^\gamma$ 接近 1 → 损失保留

γ 越大，对易分样本的压制越强

# 十、与之前课程的联系

|之前课程|在 SSD 中的体现|
|---|---|
|Day38 锚框|multibox\_prior 生成锚框、multibox\_target 生成标签、multibox\_detection 做 NMS|
|Day39 多尺度|5 个尺度特征图 \+ 5 组锚框 sizes——多尺度思想的完整实现|
|Day39 R\-CNN 对比|SSD 是单阶段（一次前向），R\-CNN 家族是两阶段|
|Day30 多通道/1×1|检测头用 3×3 卷积在特征图上滑动|
|Day34 BN|down\_sample\_blk 每层都用 BatchNorm2d|
|Day34 ResNet|「空间减半、通道翻倍」的 backbone 设计哲学|
|Day27 动态注册|setattr(self, f'blk\_\{i\}', \.\.\.\) 动态注册 5 个块|

# 十一、SSD 完整架构速查表

|组件|函数/类|作用|
|---|---|---|
|类别头|cls\_predictor|3×3 卷积，输出 num\_anchors×(C\+1\) 通道|
|框头|bbox\_predictor|3×3 卷积，输出 num\_anchors×4 通道|
|展平拼接|flatten\_pred \+ concat\_preds|多尺度预测合并|
|下采样块|down\_sample\_blk|2×Conv\+BN\+ReLU \+ MaxPool|
|基础网络|base\_net|3→16→32→64 通道|
|主网络|TinySSD|5 尺度块 \+ 5 检测头组装|
|损失|calc\_loss|CE(类别\) \+ L1(框，掩码过滤\)|
|预测|predict|softmax → multibox\_detection(NMS\)|

# 十二、本日关键记忆点

|编号|知识点|一句话|
|---|---|---|
|①|SSD 核心|单阶段 \+ 多尺度——一次前向同时预测所有锚框的类别和偏移量|
|②|多尺度实现|5 个特征图（32²→16²→8²→4²→1²）配 5 组锚框 sizes|
|③|锚框数|每个像素 4 个（2 sizes \+ 3 ratios \- 1）→ 256² 图共 5444 个|
|④|检测头|类别头输出 anchors×(C\+1\)，框头输出 anchors×4|
|⑤|多任务损失|类别用交叉熵，框用 L1（×掩码只算正样本）|
|⑥|单发含义|没有区域提议阶段，一步直接出结果——比两阶段快|
|⑦|复用 Day38|锚框生成/标签/偏移量/NMS 全是 Day38 的函数|
|⑧|Smooth L1|小误差二次\+大误差线性——对离群框鲁棒的回归损失|
|⑨|Focal Loss|调制因子 (1\-p\)ᵞ 压制易分样本——解决类别不平衡|
|⑩|预测流程|softmax → 解码偏移 → NMS → 过滤背景，得到最终框|
