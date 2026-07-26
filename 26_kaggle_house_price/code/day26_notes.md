# Day26：实战 Kaggle 比赛——预测房价

**核心主题**：完整 Kaggle 竞赛流程——数据下载与校验 → 特征工程（标准化\+独热编码）→ K 折交叉验证 → Adam 优化器 → 对数 RMSE → 生成提交文件

---

## 一、项目概览

|项目信息|说明|
|---|---|
|竞赛名称|House Prices: Advanced Regression Techniques|
|任务类型|回归——预测房屋售价|
|训练样本|1460 个|
|测试样本|1459 个|
|原始特征数|79 个（含数值和类别）|
|评估指标|**对数 RMSE**（log RMSE）|
|模型|单层线性回归 `nn.Linear(in_features, 1)`|

---

## 二、数据下载与 SHA1 校验

### 2\.1 下载配置

```python
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')  # SHA1 哈希
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

### 2\.2 下载函数（带缓存校验）

```python
def download(name, cache_dir=os.path.join('..', 'data')):
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    # 如果文件已存在，校验 SHA1
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)  # 每次读 1MB
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 校验通过，直接返回
    # 否则下载
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

|设计要点|说明|
|---|---|
|SHA1 校验|确保下载文件完整无损坏|
|分块读取|每次读 1MB，大文件也不会撑爆内存|
|缓存复用|文件已存在且校验通过 → 跳过下载|

---

## 三、数据初探

```python
train_data = pd.read_csv(download('kaggle_house_train'))
test_data  = pd.read_csv(download('kaggle_house_test'))
print(train_data.shape)  # (1460, 81)  ← 79特征 + Id + SalePrice
print(test_data.shape)   # (1459, 80)  ← 79特征 + Id（无 SalePrice）
```

训练集比测试集多一列：SalePrice（预测目标）。

---

## 四、特征工程（核心步骤）

### 4\.1 为什么合并训练集和测试集？

训练集和测试集的类别特征取值可能不完全一样。先合并、再统一处理，可以避免：

- 测试集出现训练集没见过的类别 → 独热编码维度不一致

- 标准化均值和方差计算不一致

```python
all_features = pd.concat((
    train_data.iloc[:, 1:-1],  # 去掉 Id（第0列）和 SalePrice（最后一列）
    test_data.iloc[:, 1:]      # 去掉 Id
))
# shape: (1460+1459, 79)
```

### 4\.2 数值特征：标准化（Z\-score）

```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 每列独立：减去自己的均值，除以自己的标准差
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 缺失值填 0（标准化后均值是 0，填 0 = 填均值）
```

|步骤|操作|结果|
|---|---|---|
|减去均值|$x - x.mean()$|数据中心化到 0|
|除以标准差|$/ x.std()$|各特征量纲统一|
|缺失值处理|`fillna(0)`|标准化后均值就是 0，填 0 等同于填均值|

### 4\.3 类别特征：独热编码

```python
all_features = pd.get_dummies(all_features, dummy_na=True)
```

|参数|含义|
|---|---|
|`dummy_na=True`|缺失值也作为一个类别（生成 XXX\_nan 列）|

处理后特征维度从 79 膨胀到 331（取决于类别特征有多少种取值）。

### 4\.4 转为 PyTorch 张量

```python
n_train = train_data.shape[0]  # 1460
all_features = all_features.astype(np.float32)
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features  = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels   = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),
                               dtype=torch.float32)
```

---

## 五、模型与评估指标

### 5\.1 模型：单层线性回归

```python
def get_net():
    return nn.Sequential(nn.Linear(in_features, 1))
```

特征数 = 331（预处理后），输出 = 1（房价预测值）。最简单的线性模型，但配合好的特征工程也能打得不错。

### 5\.2 评估指标：对数 RMSE（log RMSE）

竞赛官方指标。为什么用对数？

房价范围：几万 ～ 几十万（跨越多个数量级）

如果用普通 RMSE：大房子的误差主导了整体指标

如果用 log RMSE：先取对数 → 把「倍数误差」变成「加法误差」 → 大小房子一视同仁

log_rmse = sqrt( mean( (log(ŷ) - log(y))² ) ) = sqrt( mean( (log(ŷ/y))² ) )    ← 衡量"倍数误差"

```python
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # 把预测值裁剪到 ≥1（防止 log(0) → -inf）
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()
```

|步骤|操作|原因|
|---|---|---|
|数值裁剪|`clamp(..., 1, inf)` 预测值下限设为 1|防止 log\(负数\) 或 log\(0\) → \-inf|
|对数变换|对预测值和标签取对数|把乘法关系变成加法关系，均衡大小房屋误差|
|计算 RMSE|`torch.sqrt(loss(...))`|loss=MSE，开根号得到 RMSE|

---

## 六、训练函数（引入 Adam 优化器）

```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

### 6\.1 Adam vs SGD

首次使用 Adam 替代 SGD：

|对比维度|SGD|Adam|
|---|---|---|
|学习率|固定，需手动调|自适应：每个参数有自己的学习率|
|动量|可选（Momentum）|内置动量 \+ 二阶矩估计|
|对学习率敏感度|高|低，lr=5 这种 SGD 下会爆炸的值 Adam 也能跑|
|收敛速度|慢|快|
|默认选择|早期|✅ 现代深度学习常用|

🧠 Adam = Adaptive Moment Estimation。它跟踪梯度的一阶矩（均值）和二阶矩（方差），自动为每个参数调整步长。lr=5 在 SGD 中几乎不可用，但 Adam 能自适应缩放。

### 6\.2 训练循环的变化

对比 Day20 的 SGD 训练循环：

SGD 版：

    trainer = SGD\(\.\.\.\)

    trainer\.zero\_grad\(\)

    l\.mean\(\)\.backward\(\)

    trainer\.step\(\)

Adam 版：

    optimizer = Adam\(\.\.\.\)

    optimizer\.zero\_grad\(\)

    l\.backward\(\) ← 不用 \.mean\(\)，Adam 内部自适应

    optimizer\.step\(\)

⚠️ `l.backward()` 而不是 `l.mean().backward()`——因为 `nn.MSELoss()` 默认 reduction='mean'，已经对 batch 取了均值。

---

## 七、K 折交叉验证

Day23 讲过概念，今天完整落地实现。

### 7\.1 K 折数据划分

```python
def get_k_fold_data(k, i, X, y):
    """
    把数据分成 k 份，第 i 份做验证集，其余做训练集
    """
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part        # 第 i 份 → 验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part        # 第一份训练数据
        else:
            X_train = torch.cat([X_train, X_part], 0) # 后续拼接到训练集
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

图解（k=5, i=0）：

折0 [████████]  ← 验证集

    折1 [████████]  ← 训练集  ┐

    折2 [████████]  ← 训练集  │ torch\.cat 拼接

    折3 [████████]  ← 训练集  │

    折4 [████████]  ← 训练集  ┘

### 7\.2 K 折训练

```python
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                    weight_decay, batch_size)
        train_l_sum += train_ls[-1]    # 最后一轮的训练损失
        valid_l_sum += valid_ls[-1]    # 最后一轮的验证损失
        # 只画出第一折的曲线（展示趋势）
        if i == 0:
            # 绘制 train vs valid 的 log rmse 曲线
            ...
    return train_l_sum / k, valid_l_sum / k   # 返回 K 折平均
```

### 7\.3 执行 K 折验证

```python
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels,
                          num_epochs, lr, weight_decay, batch_size)
```

每折训练 100 轮，Adam 优化器 lr=5，batch\_size=64。K 折平均的验证 RMSE 才是可比的选型指标。

### 7\.4 本实验的意义

K 折交叉验证的目的是调超参数（lr、weight\_decay、num\_epochs 等），用验证集的平均误差来选最优配置，选好后再用全部训练数据训练最终模型提交。

---

## 八、全量训练与生成提交文件

K 折验证确定了超参数后，用全部训练数据（不再留验证集）训练最终模型：

```python
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels,
                        None, None,  # test_features=None → 不计算测试 loss
                        num_epochs, lr, weight_decay, batch_size)
    # 预测测试集
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    # 生成 Kaggle 提交格式
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

生成的 submission\.csv 格式：

Id	SalePrice

    1461	123456\.78

    1462	98765\.43

    \.\.\.	\.\.\.

这个文件可以直接上传到 Kaggle 提交排行榜。

---

## 九、完整流程总结

┌─────────────────────────────────────────────────────────────┐

│ 1\. 下载数据 \+ SHA1 校验                                     │

│    ↓                                                        │

│ 2\. 合并 train\+test → 统一特征工程                           │

│    ├── 数值特征：Z\-score 标准化 → fillna\(0\)                 │

│    └── 类别特征：pd\.get\_dummies(dummy\_na=True\)              │

│    ↓                                                        │

│ 3\. 转为 PyTorch Tensor                                      │

│    ↓                                                        │

│ 4\. K 折交叉验证（调超参数）                                  │

│    ├── for i in 1\.\.K:                                       │

│    │    第 i 折 = 验证集，其余 = 训练集                      │

│    │    训练 100 轮，记录 train/valid log\_rmse               │

│    └── 取 K 折平均作为评估指标                               │

│    ↓                                                        │

│ 5\. 全量训练（用全部训练数据 \+ 最优超参数）                   │

│    ↓                                                        │

│ 6\. 预测测试集 → 生成 submission\.csv → 上传 Kaggle           │

└─────────────────────────────────────────────────────────────┘

---

## 十、引入的新组件/概念

|新东西|之前用的|说明|
|---|---|---|
|Adam 优化器|SGD|自适应学习率，对 lr 不敏感|
|log RMSE|MSE / CE|回归竞赛常见指标，解决房价跨数量级问题|
|K 折交叉验证（实现）|理论概念|从概念落地到完整实现|
|torch\.clamp|未用过|裁剪预测值，防止 log(0\)|
|SHA1 文件校验|未用过|工程实践：确保下载完整性|
|pd\.get\_dummies(dummy\_na=True\)|手动填均值|更完善的类别特征处理|
|submission\.to\_csv|未用过|Kaggle 竞赛的标准输出格式|

---

## 十一、关键超参数取值

|超参数|取值|原因|
|---|---|---|
|k|5|5 折交叉验证，最常用的折数|
|num\_epochs|100|数据量不大，100 轮足够收敛|
|lr|5|Adam 可以承受高学习率|
|weight\_decay|0|本实验先行关闭，后续可以加|
|batch\_size|64|通用适中值|

---

## 十二、与之前课程联系

|之前课程|在本课的体现|
|---|---|
|Day17 pandas 预处理|fillna、get\_dummies、iloc|
|Day18 线性代数|矩阵运算在标准化 $(x-mean)/std$ 中本质是线性变换|
|Day20 线性回归|模型就是 `nn.Linear(in_features, 1)`|
|Day23 过拟合\+K折|K 折交叉验证的核心实现|
|Day24 权重衰减|weight\_decay 参数已预留，可随时开启|

---

## 十三、本日关键记忆点

|编号|知识点|一句话|
|---|---|---|
|①|训练集测试集合并预处理|先 concat，统一 z\-score 和 one\-hot，再拆开——防止编码不一致|
|②|fillna\(0\) 的巧妙之处|标准化后均值为 0，填 0 = 填均值，最简单也最合理|
|③|log RMSE|$\sqrt{\text{mean}((\log\hat{y}-\log y)^2)}$ 衡量倍数误差，大房子小房子一视同仁|
|④|clamp\(pred, 1, inf\)|防止 log\(0\) → \-inf 导致损失 NaN|
|⑤|Adam vs SGD|Adam 自适应学习率，lr=5 也能跑，现代默认选择|
|⑥|K 折实现|fold\_size = n//k，第 i 份验证，其余 torch\.cat 拼接|
|⑦|Kaggle 提交流程|预测 → test\_data['SalePrice'\] = preds → to\_csv('submission\.csv'\)|
|⑧|SHA1 校验|生产级下载代码——分块读、哈希核验、缓存复用|
