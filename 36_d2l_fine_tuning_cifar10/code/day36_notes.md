# Day36：微调 \+ Kaggle CIFAR\-10 实战

**核心主题**：微调（Fine\-tuning）——冻结预训练 backbone、替换分类头、分层学习率 → Kaggle CIFAR\-10 实战——数据重组、StepLR 调度器、生成提交文件

---

# 第一部分：微调 Fine\-tuning

## 1\.1 微调的核心思想

**问题场景**：你要做森林树种分类，但只有 200 张标注照片。从头训练 ResNet 肯定过拟合（Day23 演示过——数据少 \+ 模型大 = 灾难）。

**解决方案**：拿一个在 ImageNet（140 万张图、1000 类）上预训练好的 ResNet，把最后的分类层换成你自己的分类器（如 10 类树种），然后用你的小数据集 **继续训练**。

ImageNet 预训练模型（ResNet\-18）:

```text
├── Conv → BN → ReLU → ... → (已经学会边缘/纹理/形状/物体)
├── fc: Linear(512, 1000)   ← 学的是 ImageNet 1000 类的分类逻辑
│
└── 微调：替换 fc 层
├── 保留 backbone（冻结或小学习率）→ 通用特征提取器
└── 新 fc: Linear(512, 2) → 只训练这层（或给大 10 倍的学习率）

```

🧠 **直觉**：ResNet 的卷积层已经学会了"什么是边缘、什么是纹理、什么是形状"——这些通用视觉知识对所有图像任务都有用。你不需要重新学一遍"什么是树叶"，只需要学"这个树叶是哪个树种"。

## 1\.2 热狗数据集

一个小型二分类数据集——非常接近森林场景（小数据 \+ 二分类/多分类 \+ 迁移学习范式）：

```python
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs  = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

```

ImageFolder 自动把子文件夹名当作类别标签：

```text
hotdog/
├── train/
│   ├── hotdog/      → label = 0（按字母序排列）
│   └── not-hotdog/  → label = 1
└── test/
    ├── hotdog/
    └── not-hotdog/

```

## 1\.3 数据预处理——ImageNet 标准化

```python
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406],    # RGB 三通道的均值
    [0.229, 0.224, 0.225]     # RGB 三通道的标准差
)

```

|值|通道|含义|
|---|---|---|
|mean = [0\.485, 0\.456, 0\.406\]|R / G / B|ImageNet 数据集的每通道均值|
|std = [0\.229, 0\.224, 0\.225\]|R / G / B|ImageNet 数据集的每通道标准差|

🔑 为什么必须用 ImageNet 的统计值？ 预训练模型是在 ImageNet 上训练的——它学会了在这个标准化之后的分布上做特征提取。如果你用你自己的数据的均值和标准差，输入分布会偏移，预训练权重"不认识"这些输入，迁移效果大打折扣。用预训练模型 → 必须用预训练数据的 Normalize 参数。

## 1\.4 训练集和测试集的 Pipeline

```python
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(224),      # ① 随机裁剪到 224×224
    transforms.RandomHorizontalFlip(),      # ② 水平翻转
    transforms.ToTensor(),                  # ③ PIL → Tensor
    normalize                               # ④ ImageNet 标准化
])
test_augs = transforms.Compose([
    transforms.Resize([256, 256]),          # ① 先放大（保留更多细节）
    transforms.CenterCrop(224),             # ② 中心裁剪（不要随机！测试集固定）
    transforms.ToTensor(),                  # ③ PIL → Tensor
    normalize                               # ④ ImageNet 标准化
])

```

|对比|训练集|测试集|
|---|---|---|
|裁剪方式|RandomResizedCrop（随机）|Resize \+ CenterCrop（固定）|
|水平翻转|✅ 有|❌ 无|
|ToTensor|✅|✅|
|Normalize|用 ImageNet 均值和方差|同样用 ImageNet 均值和方差|

## 1\.5 加载预训练模型 \+ 替换分类头

```python
# 方式 1：直接加载预训练模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# pretrained_net.fc  → Linear(in_features=512, out_features=1000, bias=True)
#                           ↑ ImageNet 的 1000 类分类头

# 方式 2：替换最后一层
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
#                             ↑ 仍然保留 512 个输入特征   ↑ 改为 2 类（hotdog/not-hotdog）
nn.init.xavier_uniform_(finetune_net.fc.weight)
# 新 fc 层需要随机初始化——它没有预训练权重

```

|步骤|操作|说明|
|---|---|---|
|① 加载|resnet18(pretrained=True\)|下载 ImageNet 预训练权重|
|② 替换|\.fc = nn\.Linear(512, 2\)|把 1000 类分类头换成 2 类|
|③ 初始化|xavier\_uniform\_|新 fc 层随机初始化（旧 backbone 保持预训练值）|

## 1\.6 分层学习率——微调的关键技巧（核心）

直觉：backbone（预训练卷积层）已经有了很好的通用特征 → 只需要微调，学习率要小。新的 fc 分类头是完全随机的 → 需要从头学，学习率要大。

```python
def train_fine_tuning(net, learning_rate, ..., param_group=True):
    if param_group:
        # 分离 backbone 和 fc 层的参数
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        #  ↑ 所有 backbone 的参数
        trainer = torch.optim.SGD([
            {'params': params_1x},                          # backbone：学习率 = lr
            {'params': net.fc.parameters(),                  # fc 层：学习率 = lr × 10
             'lr': learning_rate * 10}
        ], lr=learning_rate, weight_decay=0.001)
    else:
        # 从头训练：所有参数同一学习率
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)

```

|参数组|学习率|原因|
|---|---|---|
|params\_1x（backbone）|lr（如 5e\-5）|预训练权重已经很好，只需微调|
|fc\.parameters\(\)|lr × 10（如 5e\-4）|随机初始化，需要快速收敛|

🎯 分层学习率是迁移学习的标准操作。实践中 backbone 的学习率通常设为基础学习率的 0\.1 倍。

## 1\.7 微调 vs 从头训练：对比实验

### 微调（Fine\-tuning）

```python
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(512, 2)
train_fine_tuning(finetune_net, 5e-5)   # lr=5e-5, param_group=True

```

|参数|取值|含义|
|---|---|---|
|backbone lr|5e\-5|极小的学习率——只做"微调"|
|fc lr|5e\-4|backbone 的 10 倍——快速学分类逻辑|
|pretrained|True|使用 ImageNet 预训练权重|

### 从头训练（From Scratch）

```python
scratch_net = torchvision.models.resnet18()      # pretrained=False
scratch_net.fc = nn.Linear(512, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)  # 统一 lr=5e-4

```

|参数|取值|含义|
|---|---|---|
|全部参数 lr|5e\-4|统一学习率（无分层）|
|pretrained|False|所有参数随机初始化|

### 效果对比（概念层面）

|对比维度|微调（pretrained=True）|从头训练（pretrained=False）|
|---|---|---|
|收敛速度|快（3\-5 epoch 就 OK）|慢（需要更多 epoch）|
|最终精度|高（预训练特征强）|低（小数据集训不动大模型）|
|过拟合风险|低（backbone 已稳定）|高（数据少，参数多）|
|所需数据量|少（几百张即可）|多（理想情况下数万张）|

## 1\.8 微调 = 森林 Demo 的核心技术路线

```text
森林树种分类 Demo 的技术栈：
① 数据：Kaggle 森林数据集 / 自标注树叶照片（几百张）
② 模型：torchvision.models.resnet18(pretrained=True)
③ 替换：net.fc = nn.Linear(512, num_species)
④ 分层学习率：backbone lr=1e-5, fc lr=1e-4
⑤ 数据增广：RandomResizedCrop + HorizontalFlip + ColorJitter
⑥ 测试：Resize(256) → CenterCrop(224) → Normalize(ImageNet统计值)

```

# 第二部分：Kaggle CIFAR\-10 实战

## 2\.1 数据集组织

CIFAR\-10 竞赛的原始数据格式：

```text
原始格式：
  train/            ← 训练图片（无子文件夹，需靠 CSV 标签文件）
  trainLabels.csv   ← 文件名 → 标签的映射表
  test/             ← 测试图片（无标签）

```

需要重组成 ImageFolder 可读的格式：

```text
重组后：
  train_valid_test/
  ├── train/         ← 用于训练的样本（按类别分子文件夹）
  ├── valid/         ← 用于验证的样本（每个类别取 n 张）
  ├── train_valid/   ← 训练集+验证集合并（最终全量训练用）
  └── test/unknown/  ← 测试集统一放 unknown 文件夹

```

## 2\.2 数据重组函数

### 读取 CSV 标签

```python
def read_csv_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]               # 跳过表头行
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))
    # 返回：{'1.png': 'cat', '2.png': 'dog', ...}

```

### 拆分验证集

```python
def reorg_train_valid(data_dir, labels, valid_ratio):
    # ① 找最少类别的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # ② 计算每类保留多少张做验证
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    #   例如：最少类别有 100 张，valid_ratio=0.1 → 每类留 10 张验证
    labels_count = {}
    for train_file in os.listdir(data_dir + '/train'):
        label = labels[train_file.split('.')[0]]
        # ③ 全部文件复制到 train_valid（训练+验证总集）
        copyfile(..., 'train_valid_test/train_valid/' + label + '/')
        # ④ 前 n_valid_per_label 个额外复制到 valid
        if labels_count.get(label, 0) < n_valid_per_label:
            copyfile(..., 'train_valid_test/valid/' + label + '/')
            labels_count[label] = labels_count.get(label, 0) + 1
        # ⑤ 其余复制到 train
        else:
            copyfile(..., 'train_valid_test/train/' + label + '/')

```

|变量|含义|
|---|---|
|n|样本最少的类别有多少张——保证每类都能抽出验证集|
|n\_valid\_per\_label|每类抽取多少张做验证|
|train|训练集（不含验证集部分）|
|valid|验证集（每类 n\_valid\_per\_label 张）|
|train\_valid|训练\+验证全部（用于最终训练\+预测）|

### 重组测试集

```python
def reorg_test(data_dir):
    for test_file in os.listdir(data_dir + '/test'):
        copyfile(..., 'train_valid_test/test/unknown/')
        # 所有测试文件放 unknown 文件夹——因为测试集没有标签

```

## 2\.3 训练 Pipeline

### 数据增广

```python
transform_train = transforms.Compose([
    transforms.Resize(40),                                       # 放大到 40×40
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0),         # 裁回 32×32
                                 ratio=(1.0, 1.0)),            # 保持正方形
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],             # CIFAR-10 的统计值
                         [0.2023, 0.1994, 0.2010])
])

```

|细节|说明|
|---|---|
|Resize(40\)|先放大——给 RandomResizedCrop 留出裁切空间|
|scale=(0\.64, 1\.0\)|CIFAR\-10 的图太小（32×32），裁剪比例不能太低（否则裁到的区域没有内容）|
|ratio=(1\.0, 1\.0\)|保持正方形——CIFAR\-10 本身就是正方形图片|
|Normalize 参数|用了CIFAR\-10 数据集自己的统计值（不是 ImageNet 的）——因为没用预训练模型|

📌 CIFAR\-10 和 ImageNet 的 Normalize 参数不同。因为你训练的是从头初始化的 ResNet（不是预训练的），所以用 CIFAR\-10 自己的统计值即可。

### 数据加载

```python
train_ds, train_valid_ds = [ImageFolder(..., transform=transform_train)
                             for folder in ['train', 'train_valid']]
valid_ds, test_ds       = [ImageFolder(..., transform=transform_test)
                             for folder in ['valid', 'test']]

```

四个数据集明确分工：

|数据集|用途|transform|shuffle|
|---|---|---|---|
|train\_ds|训练（调参）|train（有增广）|✅|
|valid\_ds|验证（选超参）|test（无增广）|❌|
|train\_valid\_ds|最终全量训练|train（有增广）|✅|
|test\_ds|最终预测|test（无增广）|❌|

## 2\.4 学习率调度器 StepLR

```python
trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
# lr_period = 4, lr_decay = 0.9
for epoch in range(num_epochs):
    # ... 训练循环 ...
    scheduler.step()   # ← 每个 epoch 结束后调用

```

StepLR 的效果：

```text
epoch 1-4: lr = 2e-4
epoch 5-8: lr = 2e-4 × 0.9   = 1.8e-4
epoch 9-12: lr = 2e-4 × 0.9² = 1.62e-4
epoch 13-16: lr = 2e-4 × 0.9³ = 1.458e-4
...

```

|参数|取值|含义|
|---|---|---|
|lr\_period|4|每隔 4 个 epoch 衰减一次|
|lr\_decay|0\.9|每次衰减：lr ← lr × 0\.9|
|scheduler\.step\(\)|每 epoch 调一次|内部自动计 epoch，够 period 就衰减|

🧠 为什么要衰减学习率？ 训练初期用大学习率快速接近最优解，后期用小学习率精细搜索。StepLR 是最简单经典的调度策略——按固定步长阶梯式衰减。

## 2\.5 完整训练流程

```text
第一阶段：K 折验证调参（可选，本课未做）——确定最优超参数
    │
第二阶段：用 train_ds + valid_iter 训练
    ├── 每个 epoch 输出 train loss / train acc / valid acc
    └── 用 valid_iter 监控过拟合
    │
第三阶段：用 train_valid_ds 全量训练（不再验证）
    ├── train_iter = train_valid_ds
    └── valid_iter = None（不验证）
    │
第四阶段：对 test_ds 做预测 → 生成 submission.csv

```

```python
# 第二阶段：验证训练
train(net, train_iter, valid_iter, num_epochs=20, ...)
# 第三阶段：全量训练
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs=20, ...)  # valid_iter=None
# 第四阶段：预测并生成提交文件
for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).cpu().numpy())
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('kaggle_cifar10_submission.csv', index=False)

```

第四阶段的关键细节：preds 存的是整数索引（0\-9），需要用 train\_valid\_ds\.classes[x\] 映射回类别名字符串——这是 Kaggle 提交的标准格式要求。

## 2\.6 训练超参数速查

|超参数|取值|说明|
|---|---|---|
|num\_epochs|20|CIFAR\-10 50K 样本，需较多 epoch|
|lr|2e\-4|SGD \+ momentum 的小学习率|
|wd \(weight\_decay\)|5e\-4|L2 正则化系数|
|momentum|0\.9|SGD 动量——加速收敛|
|lr\_period|4|每 4 epoch 衰减一次|
|lr\_decay|0\.9|衰减乘法因子|
|batch\_size|32 (demo\) / 128|demo 模式用小 batch|
|drop\_last|True|丢弃最后不完整 batch（防止 BN 在 batch\_size=1 时报错）|

# 第三部分：微调 vs CIFAR\-10 实战 对比

|维度|微调（热狗）|CIFAR\-10 实战|
|---|---|---|
|预训练模型|✅ pretrained=True|❌ 从头训练|
|Normalize 参数|ImageNet 统计值|CIFAR\-10 自己的统计值|
|学习率策略|分层（backbone vs fc）|统一 lr \+ StepLR 调度|
|优化器|SGD \+ weight\_decay|SGD \+ momentum \+ weight\_decay \+ scheduler|
|数据规模|小（几百张）|中（50K 张）|
|输出|训练曲线|Kaggle 提交文件|

# 第四部分：Normalize 参数速查表

|数据集|Mean (R, G, B\)|Std (R, G, B\)|何时使用|
|---|---|---|---|
|ImageNet|[0\.485, 0\.456, 0\.406\]|[0\.229, 0\.224, 0\.225\]|✅ 使用预训练模型时|
|CIFAR\-10|[0\.4914, 0\.4822, 0\.4465\]|[0\.2023, 0\.1994, 0\.2010\]|从头训练 CIFAR\-10 时|
|自定义数据集|自己算（dataset\.mean\(\)）|自己算（dataset\.std\(\)）|从头训练自己的数据时|

🔑 核心规则：用谁的预训练权重，就用谁的 Normalize 参数。因为预训练模型"习惯了"那个标准化后的数据分布。

# 第五部分：本课引入的新组件/概念

|新东西|所在文件|说明|
|---|---|---|
|微调（Fine\-tuning）|day36\_fine\_tuning\.py|用预训练模型 \+ 替换分类头 \+ 分层学习率|
|pretrained=True|fine\-tuning|加载 ImageNet 预训练权重|
|ImageNet Normalize|fine\-tuning|[0\.485,0\.456,0\.406\] / [0\.229,0\.224,0\.225\]|
|分层学习率|fine\-tuning|backbone 用小 lr，新 fc 用大 lr × 10|
|ImageFolder|两个文件|自动按子文件夹分配标签|
|StepLR 调度器|CIFAR\-10|每隔 lr\_period 个 epoch，lr ← lr × lr\_decay|
|momentum|CIFAR\-10|SGD 动量参数，默认 0\.9|
|drop\_last=True|CIFAR\-10|丢弃不完整 batch（防止 BN 报错）|
|train\_valid 合并集|CIFAR\-10|验证确定超参后合并训练\+验证做最终训练|

# 第六部分：关键代码速查

## 6\.1 微调标准模版

```python
# ① 加载预训练模型
net = torchvision.models.resnet18(pretrained=True)
# ② 替换分类头
net.fc = nn.Linear(net.fc.in_features, num_classes)
nn.init.xavier_uniform_(net.fc.weight)
# ③ 分层学习率
optimizer = torch.optim.SGD([
    {'params': [p for n, p in net.named_parameters() if 'fc' not in n]},
    {'params': net.fc.parameters(), 'lr': base_lr * 10}
], lr=base_lr, weight_decay=0.001)
# ④ Normalize 必须用 ImageNet 统计值
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

```

## 6\.2 StepLR 调度器模版

```python
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=lr_period,
                                            gamma=lr_decay)
for epoch in range(num_epochs):
    train(...)
    scheduler.step()   # ← 每 epoch 调一次

```

## 6\.3 Kaggle 提交流程模版

```python
# 预测
preds = []
for X, _ in test_iter:
    y_hat = net(X.to(device))
    preds.extend(y_hat.argmax(dim=1).cpu().numpy())
# 映射回类别名
df = pd.DataFrame({'id': range(1, len(preds)+1), 'label': preds})
df['label'] = df['label'].apply(lambda x: dataset.classes[x])
df.to_csv('submission.csv', index=False)

```

# 第七部分：本日关键记忆点

## 微调

|编号|知识点|一句话|
|---|---|---|
|①|微调 = 预训练 \+ 替换分类头|backbone 保留预训练权重（通用特征），fc 层随机初始化（任务特定）|
|②|分层学习率|backbone 用 1× lr（微量调整），fc 用 10× lr（快速收敛）|
|③|Normalize 参数|用谁的预训练权重 → 就用谁的 Normalize——ImageNet 的均值/标准差|
|④|微调 vs 从头训练|微调收敛快 \+ 精度高 \+ 抗过拟合——小数据集的唯一正确选择|

## CIFAR\-10 实战

|编号|知识点|一句话|
|---|---|---|
|⑤|数据重组|ImageFolder 要求类别分文件夹——需要把 CSV 标注 \+ 扁平的图片文件重组|
|⑥|train / valid / train\_valid|三阶段：① 训练\+验证调参 ② 验证确定最优 ③ 合并全量训练|
|⑦|StepLR|lr ← lr × gamma 每 step\_size epoch 衰减一次——训练后期的精细调优|
|⑧|momentum|SGD 的惯性项（0\.9）——加速收敛、越过局部最优|
|⑨|数据集标准化适配规则|从头训练CIFAR\-10需使用数据集自身均值方差，仅预训练模型才适配ImageNet标准化参数|
|⑩|数据增广适配策略|小尺寸图片专属增广：先放大再裁剪、限制裁切比例，避免无效裁剪，贴合32×32原图特性|
|⑪|batch 容错设置|开启drop\_last=True，丢弃最后不完整批次，规避BN层因单批次样本不足引发的报错问题|
|⑫|Kaggle提交核心逻辑|模型输出类别索引，必须映射为数据集真实类别名，才能符合竞赛提交格式规范|
