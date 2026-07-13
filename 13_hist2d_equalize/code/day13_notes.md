# Day13：二维直方图 \+ 直方图均衡化

## 一、二维直方图 — `cv.calcHist()` 多通道联合统计

### 1\. 回顾一维直方图（Day 12）

```python
hist = cv.calcHist([image], [0], None, [256], [0, 256])
```

只统计一个通道的像素分布

X 轴：像素值（0\~255），Y 轴：出现次数

**局限**：丢失了通道之间的关联信息

### 2\. 为什么需要二维直方图

|一维直方图的局限|二维直方图的优势|
|---|---|
|只看到 B/G/R 各自分布|看到两个通道之间的关联|
|无法反映颜色之间的关系|可以反映 H 和 S 的联合分布|
|红色像素多 ≠ 饱和度高|直观看出哪种色调配哪种饱和度最常见|

🌲 **林业场景**：二维直方图可以同时分析植被的色调（H）和饱和度（S）分布，区分健康植被（高饱和度绿色）和枯萎植被（低饱和度黄褐色）。

### 3\. 二维直方图的计算

```python
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
hist = cv.calcHist([hsv], [0, 1], None, [48, 48], [0, 180, 0, 256])
```

#### 参数详解

|参数|一维|二维|含义|
|---|---|---|---|
|images|[image\]|[hsv\]|输入图像列表|
|channels|[i\] 单通道|[0, 1\] 两个通道|统计 H 和 S|
|histSize|[256\]|[48, 48\]|H 分成 48 组，S 分成 48 组|
|ranges|[0, 256\]|[0, 180, 0, 256\]|H 范围 0\~180，S 范围 0\~256|

#### 关键理解

##### \(1\) 为什么是 HSV 而不是 BGR？

|色彩空间|适合二维直方图？|原因|
|---|---|---|
|HSV|✅ 最常用|H（色调）和 S（饱和度）独立，物理意义明确|
|BGR|❌|B/G/R 高度相关，二维直方图难以解读|
|Lab|✅ 可选|感知均匀，但不如 HSV 直观|

##### \(2\) 为什么 H 范围是 0\~180？

OpenCV 中 HSV 的 H 通道范围是 0\~180（不是 360°），因为 uint8 最多存 0\~255，180 是 360 的一半。实际色调 = H × 2。

##### \(3\) 为什么分成 48×48 而不是 256×256？

|bins 数|直方图尺寸|效果|
|---|---|---|
|256×256|65536 个 bin|太稀疏，可视化困难，计算量大|
|48×48|2304 个 bin|适中，足够分辨颜色，可视化成 2D 图更好看|
|32×32|1024 个 bin|粗糙，但速度快|

##### \(4\) 返回值 hist 的形状

```python
hist.shape   → (48, 48)
```

一个 48 行 × 48 列的二维数组

hist[i, j\] = H 通道落在第 i 组 \& S 通道落在第 j 组的像素个数

可以把它当作一张"灰度图"来看

### 4\. 二维直方图的可视化流程

```Plain Text
① BGR → HSV
② cv.calcHist([hsv], [0,1], ..., [48,48], ...)   → hist (48×48)
③ cv.resize(hist, (400,400))                     → 放大到 400×400，便于观察
④ cv.normalize(dst, 0, 255, NORM_MINMAX)         → 映射到 0~255
⑤ cv.applyColorMap(uint8(dst), COLORMAP_JET)     → 伪彩色显示
⑥ cv.imshow("hist", dst)                         → OpenCV 窗口显示
⑦ plt.imshow(hist, interpolation='nearest')      → matplotlib 显示（热力图）
```

### 5\. 可视化流程中的关键步骤

#### 步骤③：cv\.resize\(\) — 放大直方图

```python
dst = cv.resize(hist, (400, 400))
```

原始 hist 只有 48×48 像素 → 在屏幕上太小

放大到 400×400 → 每个 bin 对应约 8×8 个屏幕像素，清晰可见

#### 步骤④：cv\.normalize\(\) — 归一化到 0\~255

```python
cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)
```

|归一化前|归一化后|
|---|---|
|像素值是"出现次数"（可能从几百到几十万）|映射到 0\~255|
|无法直接显示（超出 0\~255 范围的部分会被截断）|可以用 imshow 正常显示|

回顾 Day 9 学的 cv\.normalize\(\)：NORM\_MINMAX 自适应地将 min→0, max→255。

#### 步骤⑤：cv\.applyColorMap\(\) — 伪彩色映射

```python
dst = cv.applyColorMap(np.uint8(dst), cv.COLORMAP_JET)
```

|项目|说明|
|---|---|
|np\.uint8(dst\)|归一化后是 float32 → 转为 uint8（0\~255 整数）|
|cv\.COLORMAP\_JET|经典彩虹色（蓝=低, 青→绿→黄→红=高）|

回顾 Day 5 学的 19 种 ColorMap。JET 是最经典的热力图配色。

#### 步骤⑥ vs ⑦：OpenCV vs matplotlib 显示对比

```python
dst = cv.applyColorMap(np.uint8(dst), cv.COLORMAP_JET)    # OpenCV 显示
cv.imshow("hist", dst)
plt.imshow(hist, interpolation='nearest')                 # matplotlib 显示
plt.title("2D Histogram")
plt.show()
```

|显示方式|interpolation='nearest' 的作用|
|---|---|
|不设置|matplotlib 默认使用平滑插值 → 直方图边缘模糊|
|'nearest'|每个 bin 用纯色块显示 → 清晰看到每个 bin 的边界|

### 6\. 如何解读二维直方图

```Plain Text
可视化结果是一张 400×400 的 JET 热力图：
X 轴 → 饱和度 S（0~255）
Y 轴 → 色调 H（0~180）
红色/黄色区域 → 出现频率高（这种 H-S 组合最常见）
蓝色/青色区域 → 出现频率低（这种 H-S 组合很少或没有）
特定位置亮度高 → 图像的色调集中在某个范围
亮度均匀分布 → 图像色彩丰富
```

## 二、直方图均衡化 — `cv.equalizeHist()`

### 1\. 问题引出

|图像类型|直方图特征|视觉效果|
|---|---|---|
|偏暗图像|像素集中在低值区（左偏）|暗部糊成一片，看不清细节|
|偏亮图像|像素集中在高值区（右偏）|亮部过曝，白成一片|
|低对比度|像素集中在中间窄范围|整体发灰，层次不清|
|理想图像|像素均匀分布 0\~255|对比度高，细节清晰|

**直方图均衡化的目标**：把任意分布的直方图拉伸成均匀分布，让图像的每个灰度级都被充分利用。

### 2\. `cv.equalizeHist()` — 核心 API

```python
image = cv.imread("D:/images/test4.png", cv.IMREAD_GRAYSCALE)  # 灰度图读取
result = cv.equalizeHist(image)
```

#### 函数签名

```python
dst = cv.equalizeHist(src)
```

|参数|要求|说明|
|---|---|---|
|src|必须是灰度图（单通道 8\-bit）|不能直接传 BGR 三通道|
|dst|输出灰度图|和输入尺寸相同|

#### ⚠️ 重要限制

```python
# ❌ 错误：直接对 BGR 图做均衡化
image = cv.imread("test.jpg")
result = cv.equalizeHist(image)    # 报错！必须是单通道

# ✅ 正确：先转灰度或分通道处理
gray = cv.imread("test.jpg", cv.IMREAD_GRAYSCALE)
result = cv.equalizeHist(gray)
```

### 3\. 均衡化的直观效果

```Plain Text
均衡化前：
直方图              图像
  |||               暗部糊成一片
  |||  (集中左侧)
  |||              亮部细节也丢失
 ─────────────────
 0        128     255

均衡化后：
直方图              图像
  |  |  |  |  |    暗部细节清晰可见
  |  |  |  |  |    (均匀分布)
  |  |  |  |  |    亮部细节也显现
 ─────────────────
 0        128     255
```

### 4\. 均衡化的原理（简化版）

```Plain Text
① 统计原图的直方图（累积分布函数 CDF）
② 对于每个原始像素值 p：
   新值 = CDF(p) × 255
   即：把累积概率映射为新的灰度值

原始:  [  30,  50,  70, 100, 200, ...]
        ↓ 映射
均衡化: [  10,  40,  80, 150, 255, ...]
→ 分布被"拉开"了，原本挤在一起的暗部像素散布到更宽的范围
```

不需要手写实现，一行 cv\.equalizeHist\(\) 搞定。但了解原理有助于理解它的局限性。

### 5\. 均衡化的局限性

|局限|说明|解决方案|
|---|---|---|
|只能处理灰度图|彩色图需要分通道处理或转色彩空间|CLAHE（自适应均衡化）|
|全局操作|整张图用同一个映射函数|CLAHE 分块均衡|
|可能过度增强噪声|暗区的噪声也被放大了|先去噪再做均衡化|
|不保持局部对比度|某些区域可能反而变差|CLAHE|

### 6\. 彩色图像怎么做均衡化

```python
# 方法一：BGR 三通道分别均衡化（简单但不推荐，可能偏色）
b, g, r = cv.split(image)
b_eq = cv.equalizeHist(b)
g_eq = cv.equalizeHist(g)
r_eq = cv.equalizeHist(r)
result = cv.merge([b_eq, g_eq, r_eq])

# 方法二：HSV 中只对 V 通道均衡化（推荐，保持色调不变）
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)
v_eq = cv.equalizeHist(v)             # 只均衡化明度通道
result_hsv = cv.merge([h, s, v_eq])
result = cv.cvtColor(result_hsv, cv.COLOR_HSV2BGR)
```

🌲 **林业应用**：对无人机航拍图做均衡化 → 增强阴影区域的植被细节 → 提高后续分割/分类的准确率。

## 三、两个函数对比

|维度|hist2d\_demo\(\)|eqhist\_demo\(\)|
|---|---|---|
|核心功能|计算并可视化 H\-S 二维直方图|灰度图直方图均衡化|
|输入|BGR 图像 → 转 HSV|灰度图（IMREAD\_GRAYSCALE）|
|核心 API|cv\.calcHist([hsv\], [0,1\], \.\.\.\)|cv\.equalizeHist\(\)|
|可视化|OpenCV \+ matplotlib 双通道显示|原图 vs 结果 并排对比|
|输出|JET 热力图（48×48→400×400）|增强后的灰度图|
|跨 Day 回顾|Day 5 (colorMap\) \+ Day 9 (normalize\) \+ Day 10 (resize\)|Day 12 (calcHist 基础\)|

## 四、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|cv\.calcHist([img\], [0,1\], \.\.\.\)|多通道联合直方图统计|Day 13 ✅|
|cv\.equalizeHist\(\)|直方图均衡化（灰度图）|Day 13 ✅|
|plt\.imshow(hist, interpolation='nearest'\)|matplotlib 显示二维直方图（无平滑）|Day 13 ✅|
|cv\.IMREAD\_GRAYSCALE|以灰度模式读取|回顾|
|cv\.COLORMAP\_JET|JET 伪彩色映射|回顾|
|cv\.normalize\(\)|归一化|回顾|

## 五、明天计划

图像卷积操作

高斯模糊
