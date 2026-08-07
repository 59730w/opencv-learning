# Day38：目标检测数据集 \+ 锚框/多尺度锚框

**核心主题**：香蕉检测数据集加载 → 锚框生成（multibox\_prior）→ IoU 计算 → 锚框分配 → 偏移量编码/解码 → 非极大值抑制 NMS → 完整预测流程

---

# 第一部分：目标检测数据集（香蕉检测数据集）

## 1\.1 数据集简介

|属性|说明|
|---|---|
|名称|banana\-detection|
|任务|检测图中的香蕉（单类别检测）|
|类别数|1（香蕉），索引为 0|
|训练集|\~1000 张|
|验证集|\~100 张|
|图片格式|PNG（`torchvision.io.read_image` 读取）|
|标签格式|CSV：`img_name, 类别, x1, y1, x2, y2`|

## 1\.2 读取数据函数

```python
def read_data_bananas(is_train=True):
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

```

|关键点|说明|
|---|---|
|set\_index('img\_name'\)|用图片名作索引，方便逐行遍历|
|read\_image|torchvision 直接读图片为 Tensor（不经过 PIL）|
|\.unsqueeze(1\)|标签 (N, 5\) → (N, 1, 5\)——第 1 维留作「对象数」|
|/ 256|坐标归一化到 [0, 1\]——目标检测的标准做法|

🧠 **标签结构**：(类别, x1, y1, x2, y2\)，其中类别 0 = 香蕉。`unsqueeze(1)` 让每个样本的标签形状是 (1, 5\)（1 个对象 × 5 个属性），这是目标检测数据集的通用格式。

## 1\.3 自定义 Dataset 类

```python
class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

```

自定义 Dataset 的三个必须方法：`__init__`（读数据）、`__getitem__`（按下标取样本）、`__len__`（样本数）。然后用 DataLoader 包一层即可批量加载。

## 1\.4 数据可视化验证

```python
batch_size, edge_size = 32, 256
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)   # torch.Size([32,3,256,256]) torch.Size([32,1,5])
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255   # C×H×W → H×W×C
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])  # 坐标×256还原像素

```

|细节|说明|
|---|---|
|batch[0\]\.shape|(32, 3, 256, 256\)——批量、通道、高、宽|
|batch[1\]\.shape|(32, 1, 5\)——批量、对象数、属性数|
|permute(0,2,3,1\)|PyTorch 的 C×H×W 转成 matplotlib 的 H×W×C|
|/255|像素值 0\~255 → 0\~1|
|label[0\][1:5\] \* 256|归一化坐标 × 图像边长 → 还原为像素坐标|

# 第二部分：锚框生成

## 2\.1 什么是锚框？

锚框（Anchor Box）是目标检测的核心思想：在图像上预置大量不同位置、不同大小、不同长宽比的候选框，然后判断每个框里有没有目标、是什么目标，并微调框的位置。

```text
在图像中每个像素位置生成多个锚框：
    ┌────┐  ┌──────────┐
    │    │  │          │
    ├────┤  ├──────────┤
    └────┘  └──────────┘
    s=0.25   s=0.5, r=2 ...

```

## 2\.2 multibox\_prior —— 生成锚框

```python
def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    ...

```

|参数|取值示例|含义|
|---|---|---|
|data|(1, 3, h, w\)|输入图像张量（只需取其 H、W）|
|sizes|[0\.75, 0\.5, 0\.25\]|锚框的相对大小（占图像比例）|
|ratios|[1, 2, 0\.5\]|宽高比（宽/高）|

**每个像素的锚框数量计算公式**：

```text
boxes_per_pixel = num_sizes + num_ratios - 1
                = 3 + 3 - 1 = 5

```

**组合逻辑**：
1\. 3 个正方形：s=0\.75/0\.5/0\.25，r=1
2\. 2 个矩形：s=0\.75，r=2 / r=0\.5
合计 5 个/像素

## 2\.3 中心点网格生成

```python
offset_h, offset_w = 0.5, 0.5          # 像素中心偏移（像素高宽为1，中心在0.5）
steps_h = 1.0 / in_height              # y 方向缩放步长
steps_w = 1.0 / in_width               # x 方向缩放步长
center_h = (torch.arange(in_height) + offset_h) * steps_h   # 归一化的中心 y 坐标
center_w = (torch.arange(in_width) + offset_w) * steps_w    # 归一化的中心 x 坐标
shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)  # 展平成所有中心点

```

|步骤|计算|结果|
|---|---|---|
|像素中心偏移|(arange \+ 0\.5\) / in\_height|每个像素中心映射到 [0,1\]|
|meshgrid|生成所有 (y, x\) 组合|共 H×W 个中心点|
|reshape(\-1\)|展平|两个长度为 H×W 的向量|

例如 H=561, W=728：shift\_x、shift\_y 各长 561×728 = 408,408。

## 2\.4 生成锚框宽高

```python
# w 的生成
w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),       # s_i × √r₀
               sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
# h 的生成
h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),       # s_i / √r₀
               sizes[0] / torch.sqrt(ratio_tensor[1:])))        # s₀ / √r_j

```

**数学公式（锚框宽高与大小 s、宽高比 r 的关系）**：

```text
宽 w = s × √r × (H/W)
高 h = s / √r

```

其中 × H/W 是处理矩形输入的修正因子（使 s 表示相对面积比例而非相对宽度）。

## 2\.5 组合成完整锚框

```python
anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(H*W, 1) / 2
#                     [(-w, -h, w, h)] 半宽半高，复制到每个中心点
out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                       dim=1).repeat_interleave(boxes_per_pixel, dim=0)
# 每个中心点重复 boxes_per_pixel 次
output = out_grid + anchor_manipulations   # 中心点 + 半宽半高 → 四角坐标
return output.unsqueeze(0)

```

|变量|形状|含义|
|---|---|---|
|anchor\_manipulations|(H×W×5, 4\)|每个锚框的 (±半宽, ±半高\)|
|out\_grid|(H×W×5, 4\)|每个锚框中心点（重复5次）|
|output|(1, H×W×5, 4\)|最终锚框：(xmin, ymin, xmax, ymax\)，已归一化|

## 2\.6 验证锚框形状

```python
img = d2l.plt.imread(r"D:\DL_code\d2l-zh\pytorch\img\catdog.jpg")
h, w = img.shape[:2]   # (561, 728)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)  # torch.Size([1, 2042040, 4])
# 561 × 728 × 5 = 2,042,040 个锚框！每个 (xmin,ymin,xmax,ymax)

```

💥 一张 561×728 的图生成了 200 万个锚框——这就是「多尺度」的含义：在每个像素位置生成多种大小和比例的先验框。

# 第三部分：IoU 交并比

## 3\.1 定义

IoU（Intersection over Union，交并比）衡量两个框的重叠程度：

```text
IoU = 交集面积 / 并集面积

```

取值范围 [0, 1\]：0 = 完全不重叠，1 = 完全重合。是评估锚框与真实框匹配度的核心指标。

## 3\.2 实现

```python
def box_iou(boxes1, boxes2):
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)    # (N1,)
    areas2 = box_area(boxes2)    # (N2,)
    # 交集左上角 = 两个框左上角取较大值
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # 交集右下角 = 两个框右下角取较小值
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # clamp(min=0)：无重叠时差值为负 → 置 0
    inter_areas = inters[:, :, 0] * inters[:, :, 1]   # 交集面积
    union_areas = areas1[:, None] + areas2 - inter_areas  # 并集面积
    return inter_areas / union_areas   # (N1, N2) IoU 矩阵
```

|形状|说明|
|---|---|
|boxes1|(N1, 4\)——N1 个锚框|
|boxes2|(N2, 4\)——N2 个真实框|
|返回|(N1, N2\)——第 i 行第 j 列 = 锚框 i 与真实框 j 的 IoU|

向量化技巧：`boxes1[:, None, :2]` 增加维度实现广播，一次算出所有框对的左上/右下角，避免双重循环。

# 第四部分：锚框分配

## 4\.1 问题

有 200 万个锚框，但只有少数包含真实目标。训练时需要一个规则：把真实边界框分配给哪些锚框作为「正样本」？

## 4\.2 算法：贪心匹配

```python
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)   # IoU 矩阵 (N_anchor, N_gt)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # -1 = 未分配（背景）
    # ① 第一轮：每个锚框取 IoU 最大的真实框，超过阈值就分配
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # ② 第二轮：保证每个真实框至少有一个锚框（贪心）
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)              # 全局最大 IoU
        box_idx = (max_idx % num_gt_boxes).long()    # 对应的真实框
        anc_idx = (max_idx / num_gt_boxes).long()    # 对应的锚框
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard            # 丢弃该列（真实框已分配）
        jaccard[anc_idx, :] = row_discard            # 丢弃该行（锚框已分配）
    return anchors_bbox_map

```

|轮次|规则|
|---|---|
|第一轮|锚框的最大 IoU ≥ 0\.5 → 分配该真实框|
|第二轮|每个真实框必须有一个锚框（从 IoU 最大的开始贪心匹配）|
|未分配|anchors\_bbox\_map = \-1 → 该锚框是背景|

🎯 两轮分配保证：① 高质量锚框被标记为正样本；② 每个真实目标至少被一个锚框「认领」（避免漏检）。

# 第五部分：偏移量编码

## 5\.1 为什么预测偏移量而不是直接预测坐标？

锚框位置是固定的，直接回归坐标数值范围大、难收敛。更好的做法：预测「锚框到真实框的相对偏移」。

## 5\.2 offset\_boxes —— 编码（锚框 → 真实框的偏移）

```python
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = d2l.box_corner_to_center(anchors)          # 锚框中心表示
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)  # 真实框中心表示
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    #         ^^  中心偏移 / 锚框宽高 × 10
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    #        ^  log(真实宽高/锚框宽高) × 5
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

```

**数学公式**：

```text
Δx = 10 × (cx_true − cx_anchor) / w_anchor
Δy = 10 × (cy_true − cy_anchor) / h_anchor
Δw = 5 × ln(w_true / w_anchor)
Δh = 5 × ln(h_true / h_anchor)

```

|因子|作用|
|---|---|
|除以锚框宽高|归一化——偏移量与锚框大小无关|
|×10|放大中心偏移（实际偏差通常很小）|
|×5 \+ log|宽高用对数比（保证宽高恒为正），乘以 5 缩放|

## 5\.3 offset\_inverse —— 解码（预测偏移 → 真实框）

```python
def offset_inverse(anchors, offset_preds):
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    return d2l.box_center_to_corner(pred_bbox)

```

**编码逆运算公式**：

```text
cx = Δx × w_anchor / 10 + cx_anchor
cy = Δy × h_anchor / 10 + cy_anchor
w  = exp(Δw / 5) × w_anchor
h  = exp(Δh / 5) × h_anchor

```

编码用于训练标签，解码用于预测输出——两者互为逆运算。

# 第六部分：multibox\_target —— 完整标记锚框

## 6\.1 输出三样东西

```python
def multibox_target(anchors, labels):
    ...
    return (bbox_offset, bbox_mask, class_labels)

```

|输出|形状|含义|
|---|---|---|
|bbox\_offset|\(batch, num\_anchors×4\)|每个锚框到其分配真实框的偏移量|
|bbox\_mask|\(batch, num\_anchors×4\)|掩码：正样本锚框=1，背景锚框=0（屏蔽背景的偏移损失）|
|class\_labels|\(batch, num\_anchors\)|每个锚框的类别（0=背景，1\+=真实类别索引\+1）|

## 6\.2 关键逻辑

```python
# 类别标签初始化全 0（背景）
class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
# 找到被分配了真实框的锚框
indices_true = torch.nonzero(anchors_bbox_map >= 0)
bb_idx = anchors_bbox_map[indices_true]
class_labels[indices_true] = label[bb_idx, 0].long() + 1   # 类别+1（0留给背景）
assigned_bb[indices_true] = label[bb_idx, 1:]               # 取真实框坐标
# 偏移量计算 × 掩码（背景锚框偏移量被置 0）
offset = offset_boxes(anchors, assigned_bb) * bbox_mask

```

类别标签用 \+1：0 表示背景，真实类别从 1 开始编号——这样背景和正样本天然区分。

# 第七部分：非极大值抑制 NMS

## 7\.1 为什么要 NMS？

一张图上会生成上百万个锚框，预测时很多锚框会重叠地指向同一个目标。NMS 的目标：去掉重复的框，只保留最好的那个。

## 7\.2 实现

```python
def nms(boxes, scores, iou_threshold):
    B = torch.argsort(scores, dim=-1, descending=True)   # 按置信度降序
    keep = []
    while B.numel() > 0:
        i = B[0]                                          # 取当前最高分框
        keep.append(i)
        if B.numel() == 1:
            break
        # 计算该框与其余框的 IoU
        iou = box_iou(boxes[i].reshape(-1, 4), boxes[B[1:]].reshape(-1, 4)).reshape(-1)
        # 保留与最高分框 IoU ≤ 阈值 的框（抑制掉重叠过高的）
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]                                   # 更新候选集
    return torch.tensor(keep, device=boxes.device)

```

**算法流程**：

```text
① 所有预测框按置信度从高到低排序
② 取置信度最高的框 → 加入 keep 列表
③ 计算它与剩余框的 IoU
④ 删除与该框 IoU > 阈值 的框（重叠太严重的冗余框）
⑤ 回到 ②，直到没有剩余框

```

📌 iou\_threshold=0\.5：两个框 IoU 超过 0\.5 就认为它们是「同一个目标的两个预测」，只保留分数高的。

# 第八部分：multibox\_detection —— 完整预测流程

```python
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    ...
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # ① 取每个锚框预测概率最高的类别和置信度（排除背景类 0）
        conf, class_id = torch.max(cls_prob[1:], 0)
        # ② 用预测偏移解码出真实框坐标
        predicted_bb = offset_inverse(anchors, offset_pred)
        # ③ NMS 去重
        keep = nms(predicted_bb, conf, nms_threshold)
        # ④ 未被保留的框类别设为 -1（背景/被抑制）
        non_keep = uniques[counts == 1]
        class_id[non_keep] = -1
        # ⑤ 低置信度（< pos_threshold）也设为背景
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        # ⑥ 组装输出 (类别, 置信度, x1, y1, x2, y2)
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)

```

**输出格式**：

```text
output[0] 形状: (num_anchors, 6)
每行: [类别, 置信度, x1, y1, x2, y2]
类别 = -1 → 背景/被抑制（不画）
类别 = 0  → dog
类别 = 1  → cat

```

**完整预测流程总结**：

```text
预测输出（每个锚框：类别概率 + 偏移量）
    │
    ├─ ① 取最大概率类别作为锚框类别，该概率作为置信度
    ├─ ② 偏移量解码 → 预测边界框坐标
    ├─ ③ NMS：按置信度排序，抑制 IoU 过高的冗余框
    ├─ ④ 被抑制的框 → 背景（-1）
    ├─ ⑤ 置信度低于 pos_threshold → 背景（-1）
    └─ ⑥ 输出 (类别, 置信度, 坐标) → 绘制在图上

```

# 第九部分：目标检测流程全景图（Day37\~38）

```text
Day37：边界框表示（角点/中心）→ 画框可视化
        │
Day38：数据集（香蕉检测）→ 锚框生成（200万+）→ IoU → 锚框分配 → 偏移量
        │
        训练（预测类别 + 偏移量）  ← 由网络完成
        │
        NMS 去重 → 输出最终检测结果

```

```text
训练时： 真实框 + 锚框 → IoU → 分配 → 类别标签 + 偏移量标签 → 监督网络
预测时： 网络输出类别概率 + 偏移量 → 解码坐标 → NMS → 最终框

```

# 第十部分：本日关键记忆点

|编号|知识点|一句话|
|---|---|---|
|①|锚框数量|每像素 num\_sizes \+ num\_ratios \- 1 个——561×728 图 → 200 万锚框|
|②|锚框生成|中心点网格 \+ 多尺寸多比例，坐标归一化到 [0,1\]|
|③|IoU|交并比——衡量两个框重叠程度，是匹配/评估的核心指标|
|④|锚框分配|两轮贪心：阈值分配 \+ 每个真实框必有锚框|
|⑤|偏移量编码|预测相对偏移（Δx, Δy, Δw, Δh）而非绝对坐标|
|⑥|偏移量公式|Δxy 除以锚框宽高×10；Δwh 用 log 比×5|
|⑦|解码|编码的逆运算：exp(Δw/5\)×w\_anchor 恢复真实宽高|
|⑧|NMS|按置信度排序，抑制 IoU \> 阈值的重叠框——去重|
|⑨|输出格式|(类别, 置信度, x1, y1, x2, y2\)，\-1 = 背景|
|⑩|香蕉数据集|单类别检测，标签 (N,1,5\)，坐标归一化，方便入门|
