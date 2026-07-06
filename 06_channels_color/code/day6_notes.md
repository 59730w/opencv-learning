# Day6：通道分离与合并 \+ 色彩空间转换与掩码提取

## 一、通道分离与合并

### 1\. 通道的本质

BGR 彩色图像 = 三个独立的二维数组堆叠在一起：

```Plain Text
BGR 图像 (h, w, 3)
├── B 通道 (h, w) — 蓝色分量
├── G 通道 (h, w) — 绿色分量
└── R 通道 (h, w) — 红色分量
```

每个通道都是一个**单通道灰度图**，像素值 0\~255 表示该颜色分量的强度。

### 2\. `cv.split()` — 分离通道

```python
mv = cv.split(b1)
```

|项目|说明|
|---|---|
|输入|BGR 三通道图像|
|输出|包含 3 个数组的元组：\(B通道, G通道, R通道\)|
|mv[0\]|B 通道（蓝色分量）|
|mv[1\]|G 通道（绿色分量）|
|mv[2\]|R 通道（红色分量）|

#### 快速提取单个通道的另一种方式

```python
b1[:, :, 2]    # 提取 R 通道（索引 2 = R）
```

|索引|通道|颜色|
|---|---|---|
|[:, :, 0\]|B|蓝色|
|[:, :, 1\]|G|绿色|
|[:, :, 2\]|R|红色（⚠️ 不是 BGR 顺序吗？索引反的）|

⚠️**重要澄清**：OpenCV 的图像是 [行, 列, 通道\]，通道顺序是 B=0, G=1, R=2。所以 b1[:,:,2\] 提取的是 R 通道。

### 3\. 修改单通道 → cv\.merge\(\) 合并

```python
mv = cv.split(b1)        # 分离
mv[0][:, :] = 255        # 把 B 通道的所有像素设为 255（最大蓝色）
result = cv.merge(mv)    # 合并回去
```

#### 效果解析

```Plain Text
原始 B 通道值：各种 0~255 的值
修改后 B 通道：全部变 255
合并后 → 整张图的 B 分量全满 → 图片偏蓝
```

|操作|效果|
|---|---|
|mv[0\][:,:\] = 255|B 通道拉满 → 整张图蒙上蓝色调|
|mv[1\][:,:\] = 255|G 通道拉满 → 整张图蒙上绿色调|
|mv[2\][:,:\] = 255|R 通道拉满 → 整张图蒙上红色调|
|mv[0\][:,:\] = 0|B 通道归零 → 去除蓝色分量|

🌲 **林业场景应用**：把 G 通道（绿色）拉满，可以增强植被区域的显示效果；把 R 通道拉满，可以突出土壤/枯枝区域。

### 4\. cv\.mixChannels\(\) — 通道重排（重点！）

```python
dst = np.zeros(b1.shape, dtype=np.uint8)
cv.mixChannels([b1], [dst], fromTo=[2,0, 1,1, 0,2])
```

#### 函数签名

```python
cv.mixChannels([src1, src2, ...], [dst1, dst2, ...], fromTo)
```

#### fromTo 参数详解

fromTo 是一个成对出现的列表，格式：[源索引, 目标索引, 源索引, 目标索引, \.\.\.\]

|对|fromTo 值|含义|
|---|---|---|
|第1对|2, 0|源图的通道2（R）→ 目标图的通道0（B）|
|第2对|1, 1|源图的通道1（G）→ 目标图的通道1（G）|
|第3对|0, 2|源图的通道0（B）→ 目标图的通道2（R）|

#### 效果

```Plain Text
原始 BGR:  [B, G, R]
混合后:    [R, G, B]   ← B 和 R 互换位置了！
```

这本质上就是 BGR → RGB 转换的另一种实现方式，相比 cv\.cvtColor(img, COLOR\_BGR2RGB\)，mixChannels 更灵活，可以任意重排通道。

#### cv\.split/merge vs cv\.mixChannels

|方式|优点|缺点|
|---|---|---|
|split → 修改 → merge|直观、易理解|三步操作，split 会复制内存|
|mixChannels|一步到位，内存高效|fromTo 参数不够直观|

## 二、色彩空间转换 \+ 颜色掩码提取

### 1\. 回顾 HSV 色彩空间

```python
hsv = cv.cvtColor(b1, cv.COLOR_BGR2HSV)
```

|分量|范围|含义|
|---|---|---|
|H（色调）|0\~180（OpenCV中）|颜色种类：0=红, 60=绿, 120=蓝|
|S（饱和度）|0\~255|颜色纯度：0=灰, 255=最鲜艳|
|V（明度）|0\~255|亮度：0=黑, 255=最亮|

### 2\. cv\.inRange\(\) — 颜色范围过滤（核心！）

```python
mask = cv.inRange(hsv, (0, 0, 221), (180, 30, 255))
```

#### 函数签名

```python
mask = cv.inRange(src, lowerb, upperb)
```

|参数|含义|本例取值|
|---|---|---|
|src|输入图像（通常是 HSV）|hsv|
|lowerb|颜色下限 (H\_min, S\_min, V\_min\)|(0, 0, 221\)|
|upperb|颜色上限 (H\_max, S\_max, V\_max\)|(180, 30, 255\)|
|返回值|二值掩码图像：在范围内=255（白色），不在=0（黑色）|\-|

#### 本例中筛选的是什么颜色？

```Plain Text
H: 0 ~ 180       → 所有色调（不限制颜色）
S: 0 ~ 30        → 低饱和度（接近灰色/白色）
V: 221 ~ 255     → 高亮度
```

这是在筛选 高亮且低饱和度的像素——换句话说，就是白色/亮灰色区域。

### 3\. 掩码取反 \+ 抠图

```python
cv.bitwise_not(mask, mask)                          # 掩码取反
result = cv.bitwise_and(b1, b1, mask=mask)          # 用掩码提取区域
```

#### 为什么要取反？

```Plain Text
inRange 选出的：白色/高亮区域（背景，想剔除的）
bitwise_not 后：黑色区域变白 → 选中了「非白色」区域（前景，想保留的）
```

#### 完整流程

```Plain Text
原图 → cvtColor(BGR→HSV) → inRange(选出高亮白色区域) → bitwise_not(取反，选中前景)
                                                              ↓
                                            bitwise_and(原图, 原图, mask) → 扣出前景
```

|步骤|操作|结果|
|---|---|---|
|①|cv\.cvtColor(BGR→HSV\)|HSV 图像|
|②|cv\.inRange(hsv, low, high\)|白色区域 = 255，其余 = 0|
|③|cv\.bitwise\_not(mask\)|白色区域变 0（丢弃），其余变 255（保留）|
|④|cv\.bitwise\_and(b1, b1, mask=mask\)|掩码为 0 的像素变黑，掩码为 255 的保持原样|

#### bitwise\_and(src1, src2, mask\) 的工作原理

```Plain Text
对于每个像素：
  if mask 像素 == 0:   结果像素 = 0（黑色）
  if mask 像素 != 0:   结果像素 = src1 & src2
```

🌲 这是从图像中提取特定颜色区域的标准方法，林业应用中常用于提取绿色植被、去除天空/云层背景、分离不同树种等。

## 三、完整的颜色提取流程（标准模板）

```python
# 1. 读取原图
bgr = cv.imread("image.jpg")
# 2. 转为 HSV
hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
# 3. 设定目标颜色的 HSV 范围
lower = (H_min, S_min, V_min)
upper = (H_max, S_max, V_max)
# 4. 生成掩码
mask = cv.inRange(hsv, lower, upper)
# 5. 可选：掩码取反（保留非目标色区域）
mask = cv.bitwise_not(mask)
# 6. 用掩码提取
result = cv.bitwise_and(bgr, bgr, mask=mask)
```

## 四、两个函数对比

|维度|channels\_split\_merge\_demo\(\)|color\_space\_demo\(\)|
|---|---|---|
|核心操作|通道分离(split\) \+ 修改 \+ 合并(merge\) \+ 重排(mixChannels\)|HSV 转换 \+ inRange 颜色筛选 \+ 掩码抠图|
|输入|test1\.png|test3\.png|
|关键 API|split，merge，mixChannels|cvtColor，inRange，bitwise\_not，bitwise\_and|
|输出效果|B/G/R 通道重组、蓝色增强|去除高亮白色背景，保留前景|
|林业应用|通道增强（突出植被 G 通道）|颜色筛选（提取绿色植被，去除天空背景）|

## 五、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|cv\.split\(\)|分离 BGR 三通道|Day 6 ✅|
|cv\.merge\(\)|合并三通道为 BGR 图像|Day 6 ✅|
|cv\.mixChannels\(\)|灵活重排通道（fromTo 参数）|Day 6 ✅|
|cv\.inRange\(\)|颜色范围筛选，生成二值掩码|Day 6 ✅|
|cv\.bitwise\_and(src1, src2, mask\)|带掩码的按位与（掩码抠图）|Day 6 ✅|
|cv\.bitwise\_not\(\)|掩码取反（回顾）|Day 4|

## 六、明天计划

图像像素值的统计

图像几何形状的绘制
