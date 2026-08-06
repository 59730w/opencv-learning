# Day37：Kaggle 狗品种识别实战 \+ 目标检测与边界框

**核心主题**：Kaggle ImageNet Dogs 实战（冻结式微调、120类分类、softmax 概率提交）→ 目标检测入门（边界框的两种表示法、转换、可视化）

---

# 第一部分：Kaggle 狗品种识别实战

## 1\.1 项目概览

|项目信息|说明|
|---|---|
|竞赛名称|Dog Breed Identification \(ImageNet Dogs\)|
|任务类型|**120 类**细粒度图像分类（狗品种识别）|
|数据形式|图片文件 \+ `labels.csv`（文件名 → 品种标签）|
|关键技术|微调 ResNet\-34（冻结 backbone）|
|评估指标|多分类对数损失（CrossEntropyLoss）|
|提交格式|每张测试图输出 120 类概率分布|

这是你 Day36 学的「微调」技术在真实竞赛中的完整应用。和热狗二分类不同，这是 **120 类**的细粒度分类——狗品种之间的差异极小（哈士奇 vs 阿拉斯加），正是细粒度分类的经典难题。

## 1\.2 与 Day36 的相同点（标准流程复用）

|步骤|与 Day36 相同|
|---|---|
|数据重组|相同的 `reorg_dog_data` → `read_csv_labels` \+ `reorg_train_valid` \+ `reorg_test`|
|数据增广|`RandomResizedCrop(224)` \+`RandomHorizontalFlip` \+ `ColorJitter`|
|测试集 Pipeline|`Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize(ImageNet)`|
|数据集划分|train / valid / train\_valid / test 四个数据集|
|学习率调度|`StepLR(lr_period=2, lr_decay=0.9)`|
|训练两阶段|① train\+valid 验证 → ② train\_valid 全量训练|

🎯 Day36 已经建立了完整的竞赛流程模板，今天的代码是**套模板 \+ 换模型 \+ 改输出格式**。

## 1\.3 与 Day36 的不同点——三种微调策略的演进

Day36 热狗用了「**backbone 保留 \+ 替换 fc \+ 分层学习率**」。今天的狗品种识别用了**更激进的冻结式微调**：

```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(
        nn.Linear(1000, 256),    # 1000 → 256
        nn.ReLU(),
        nn.Linear(256, 120))     # 256 → 120（120个狗品种）
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数：backbone 完全不动！
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

```

|对比|Day36 热狗|Day37 狗品种|
|---|---|---|
|预训练模型|ResNet\-18|ResNet\-34（更深）|
|分类头|Linear(512, 2\)|两层 MLP Linear(1000,256\) → Linear(256,120\)|
|backbone 训练|小学习率微调（lr×1）|完全冻结（requires\_grad=False）|
|只训练什么|backbone \+ fc|只有 output\_new 两层|

## 1\.4 为什么这里冻结 backbone？

```text
Day36 热狗： 只换最后一层，backbone 微调 → 因为输出从1000→2，分类头小，backbone还有学习空间
Day37 狗：   换两层 MLP，backbone 完全冻结 → 因为 ResNet-34 的 1000 维特征对狗品种已足够有区分度，
             冻结可以省显存、防过拟合、加速训练（不需要给 backbone 算梯度）

```

🧠**判断依据**：如果预训练特征（ImageNet 1000 维向量）已经足够区分你的任务（狗品种），就冻结 backbone，只训分类头。如果任务域差异大（如医疗影像 vs 自然图像），才需要解冻微调。

## 1\.5 nn\.Sequential\(\) 空容器 \+ 动态添加属性

```python
finetune_net = nn.Sequential()          # 空 Sequential
finetune_net.features = resnet34(...)   # 动态添加 features 属性
finetune_net.output_new = nn.Sequential(...)  # 动态添加 output_new 属性

```

这里利用了 nn\.Module 的 **\_\_setattr\_\_** 魔法（Day27 学过）——赋值的子模块会自动注册，能被 net\.parameters\(\) 收集。

## 1\.6 只训练可训练参数

```python
trainer = torch.optim.SGD((param for param in net.parameters()
                           if param.requires_grad),   # ← 只取 requires_grad=True 的参数
                          lr=lr, momentum=0.9, weight_decay=wd)

```

🔑 冻结的 backbone 参数 `requires_grad=False`，不会出现在优化器中——既不计算梯度，也不更新。优化器只管理 output\_new 的 1000×256 和 256×120 两层参数。

## 1\.7 训练损失函数与验证

```python
loss = nn.CrossEntropyLoss(reduction='none')
def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')

```

|细节|说明|
|---|---|
|reduction='none'|逐样本损失向量|
|evaluate\_loss|验证集用 平均损失 而非准确率——因为竞赛指标是对数损失|
|\.to('cpu'\)|最终指标搬回 CPU 显示|

📌 注意：这个任务用 loss 而非 accuracy 作为监控指标——因为 Kaggle 的提交排名依据是对数损失，训练目标与评估目标必须一致。

## 1\.8 预测与提交文件——概率分布格式

与其他竞赛不同，狗的品种识别要求输出 120 类概率分布（不是单个类别索引）：

```python
preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())   # 每行是一个 120 维概率向量
ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('kaggle_dog_submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')   # 表头：id + 120个品种名
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')               # 每行：id + 120个概率

```

提交文件格式：

```text
id,affenpinscher,afghan_hound,african_hunting_dog,...(120列)
000bec180eb18c7604dcecc8fe0dba07,0.0012,0.0003,0.0008,...
...

```

|对比|CIFAR\-10 \(Day36\)|狗品种 \(Day37\)|
|---|---|---|
|输出内容|类别索引（argmax）|softmax 概率分布|
|输出维度|1 个数|120 个数|
|提交格式|id,label|id,class1\_prob,class2\_prob,\.\.\.|
|评估方式|准确率|对数损失（需要概率）|

🔑 为什么输出概率而不是类别？ 多分类对数损失（log loss）直接惩罚「预测概率与真实标签的偏差」——只有给出概率分布才能计算损失。而且当模型不确定时，概率分布比硬性类别更有信息量。

## 1\.9 完整流程总结

```text
① 下载 dog_tiny 数据集（demo=True 小样本 / False 完整竞赛数据）
② 数据重组：CSV 标签 → 按类别分文件夹
③ 数据增广（train）/ 中心裁剪（test）
④ 微调 ResNet-34：冻结 backbone + 新两层分类头（1000→256→120）
⑤ 优化器只含 output_new 参数，StepLR 调度
⑥ 两阶段训练：train+valid 验证 → train_valid 全量
⑦ softmax 概率 → 写 120 列概率 CSV → 上传 Kaggle
```

# 第二部分：目标检测与边界框

## 2\.1 目标检测 vs 图像分类

|任务|要回答的问题|输出|
|---|---|---|
|图像分类|「这是什么？」|一个类别标签|
|目标检测|「这是什么？」\+「在哪里？」|类别标签 \+ 边界框位置|

从今天开始，你的森林应用从「这是什么树」进化到「树在哪里 \+ 是什么树」——这就是森林巡检、病虫害监测等实际林业场景的需求。

## 2\.2 边界框（Bounding Box）

边界框是目标检测的最基本概念——用一个矩形框住目标。

加载一张同时有猫和狗的图片：

```python
img = d2l.plt.imread(r"D:\DL_code\d2l-zh\pytorch\img\catdog.jpg")
d2l.plt.imshow(img)

```

手工标注的两个边界框：

```python
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
#                     [x1,   y1,   x2,    y2   ]      [x1,   y1,   x2,   y2   ]

```

## 2\.3 边界框的两种表示法

|表示法|内容|本课代码|
|---|---|---|
|角点表示法 (corner\)|(x1, y1, x2, y2\) —— 左上角 \+ 右下角|dog\_bbox|
|中心表示法 (center\)|(cx, cy, w, h\) —— 中心点 \+ 宽 \+ 高|转换后|

### 角点 → 中心

```python
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2      # 中心 x = (左 + 右) / 2
    cy = (y1 + y2) / 2      # 中心 y = (上 + 下) / 2
    w  = x2 - x1            # 宽 = 右 - 左
    h  = y2 - y1            # 高 = 下 - 上
    return torch.stack((cx, cy, w, h), axis=-1)

```

### 中心 → 角点

```python
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w       # 左 = 中心x - 半宽
    y1 = cy - 0.5 * h       # 上 = 中心y - 半高
    x2 = cx + 0.5 * w       # 右 = 中心x + 半宽
    y2 = cy + 0.5 * h       # 下 = 中心y + 半高
    return torch.stack((x1, y1, x2, y2), axis=-1)

```

### 转换验证

```python
boxes = torch.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
# 全 True ← 来回转换无损

```

|转换公式|角点→中心|中心→角点|
|---|---|---|
|中心 x|(x1\+x2\)/2|cx \- w/2 → x1|
|中心 y|(y1\+y2\)/2|cy \- h/2 → y1|
|宽/高|x2\-x1, y2\-y1|cx \+ w/2 → x2|

🧠 为什么需要两种表示？ 标注时用角点直观（鼠标拖框），而后续的锚框、IoU 计算、YOLO 训练目标常用中心\+宽高表示（更利于回归预测）。两者互转是目标检测的基础操作。

## 2\.4 在图上绘制边界框

```python
def bbox_to_rect(bbox, color):
    """把 (x1,y1,x2,y2) 转成 matplotlib 的 Rectangle 对象"""
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]),           # 左上角 (x1, y1)
        width=bbox[2]-bbox[0],           # 宽度 = x2 - x1
        height=bbox[3]-bbox[1],          # 高度 = y2 - y1
        fill=False,                      # 不填充，只画边框
        edgecolor=color,                 # 边框颜色
        linewidth=2)                     # 线宽
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))   # 狗 → 蓝框
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))    # 猫 → 红框

```

📌 matplotlib 的 Rectangle 需要 (左上角坐标, 宽, 高\)，而边界框是 (x1, y1, x2, y2\)——bbox\_to\_rect 做这个格式转换。

# 第三部分：本日关键记忆点

## Kaggle 狗品种

|编号|知识点|一句话|
|---|---|---|
|①|冻结式微调|backbone requires\_grad=False，只训练新分类头——省显存 \+ 防过拟合|
|②|动态注册子模块|nn\.Sequential\(\) 空容器 \+ \.features = \.\.\. / \.output\_new = \.\.\.|
|③|只训可训练参数|优化器传入 (p for p in net\.parameters\(\) if p\.requires\_grad\)|
|④|细粒度分类|120 类狗品种差异极小——比 CIFAR\-10 难得多|
|⑤|概率提交|输出 120 维 softmax 概率（非 argmax）——因为评估用 log loss|
|⑥|loss 作监控指标|竞赛指标是对数损失，训练监控也用 loss 而非 accuracy|

## 目标检测

|编号|知识点|一句话|
|---|---|---|
|⑦|检测 vs 分类|分类只答「是什么」，检测还要答「在哪里」——用边界框定位|
|⑧|角点表示|(x1,y1,x2,y2\) 左上\+右下——标注直观|
|⑨|中心表示|(cx,cy,w,h\) 中心\+宽高——适合回归预测|
|⑩|互转公式|角点→中心：cx=(x1\+x2\)/2；中心→角点：x1=cx\-w/2|
|⑪|matplotlib 画框|Rectangle(xy=左上角, width, height\)——需要格式转换|
