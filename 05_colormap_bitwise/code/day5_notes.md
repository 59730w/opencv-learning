# Day5：OpenCV 颜色表操作 \+ 图像像素逻辑运算

## 一、OpenCV 自带颜色表（Colormap）

### 1\. 什么是颜色表

颜色表（Colormap）是一种**伪彩色映射**：把灰度图的像素值（0\~255）映射到预先定义好的颜色梯度上，从而把灰度图变成彩色图。

#### 应用场景（与森林监测相关）

|场景|说明|
|---|---|
|热力图显示|用颜色反映植被指数（如 NDVI）的高低|
|深度图可视化|把距离信息映射成颜色，区分远近|
|数据增强|生成不同风格的训练样本|
|异常区域标注|用醒目颜色高亮异常区域|

### 2\. `cv.applyColorMap()` 核心 API

```python
dst = cv.applyColorMap(image, colormap_type)
```

|参数|说明|
|---|---|
|image|输入图像（通常为灰度图，也可以是 BGR 图）|
|colormap\_type|颜色表类型，如 cv\.COLORMAP\_JET|
|返回值 dst|应用颜色表后的伪彩色图像（BGR 三通道）|

### 3\. OpenCV 内置的 19 种颜色表

|索引|颜色表常量|风格|适用场景|
|---|---|---|---|
|0|COLORMAP\_AUTUMN|秋天色（红→黄）|温度图|
|1|COLORMAP\_BONE|骨白色（黑→灰白）|X光风格|
|2|COLORMAP\_JET|经典彩虹色（蓝→青→绿→黄→红）|最常用，热力图首选|
|3|COLORMAP\_WINTER|冬色（蓝→绿）|低温场景|
|4|COLORMAP\_RAINBOW|彩虹色|彩色显示|
|5|COLORMAP\_OCEAN|海洋色（蓝→白）|海洋/水体|
|6|COLORMAP\_SUMMER|夏色（绿→黄）|植被覆盖|
|7|COLORMAP\_SPRING|春色（粉→黄）|柔和显示|
|8|COLORMAP\_COOL|冷色调（青→粉）|科技感|
|9|COLORMAP\_PINK|粉色系（黑→粉→白）|柔和显示|
|10|COLORMAP\_HOT|热色（黑→红→黄→白）|热力图|
|11|COLORMAP\_PARULA|感知均匀（蓝→黄）|科学可视化|
|12|COLORMAP\_MAGMA|岩浆色（黑→紫→红→黄）|感知均匀，替代 JET|
|13|COLORMAP\_INFERNO|地狱色（黑→红→黄→白）|感知均匀|
|14|COLORMAP\_PLASMA|等离子色（黑→紫→黄）|感知均匀|
|15|COLORMAP\_VIRIDIS|翠绿（紫→蓝→绿→黄）|色盲友好，科学可视化首选|
|16|COLORMAP\_CIVIDIS|蓝黄渐变|色盲友好|
|17|COLORMAP\_TWILIGHT|暮光色|特殊效果|
|18|COLORMAP\_TWILIGHT\_SHIFTED|暮光偏移版|特殊效果|

💡 MAGMA、INFERNO、PLASMA、VIRIDIS、CIVIDIS 是 matplotlib 同款色表，感知均匀，适合科学论文配图。

### 4\. 轮播颜色表的巧妙实现

```python
colormap = [cv.COLORMAP_AUTUMN, cv.COLORMAP_BONE, ..., cv.COLORMAP_TWILIGHT_SHIFTED]
index = 0
while True:
    dst = cv.applyColorMap(image, colormap[index % 19])   # 取模实现循环
    index += 1
    cv.imshow("color style", dst)
    c = cv.waitKey(500)   # 每 500ms 切换一次
    if c == 27:
        break
```

#### 关键技巧：index % 19 取模循环

```Plain Text
index:  0   1   2  ...  18   19   20 ...
%19:    0   1   2  ...  18   0    1  ...
```

- 19 是 ColorMap 列表的长度

- 取模让索引在 0\~18 之间无限循环

- waitKey\(500\) 每 0\.5 秒自动切换，形成轮播效果

这种 列表[index % 列表长度\] 的循环播放在项目中很实用，比如轮播不同滤波效果对比。

## 二、图像像素的逻辑（位）运算

### 1\. 为什么需要位运算

图像 = NumPy 数组，每个像素值 = 一个 8 位整数（0\~255）。位运算就是对这个整数的每个二进制位做 AND/OR/NOT/XOR。

#### 位运算在图像处理中的典型应用

|应用|使用运算|说明|
|---|---|---|
|掩码提取|bitwise\_and|保留 ROI，屏蔽其余区域|
|图像叠加|bitwise\_or|合成两张图|
|图像取反|bitwise\_not|反色效果（Day 4 已实现）|
|水印去除|bitwise\_xor|XOR 两次恢复原图|

### 2\. 核心 API

```python
dst1 = cv.bitwise_and(src1, src2)    # 按位与
dst2 = cv.bitwise_or(src1, src2)     # 按位或
dst3 = cv.bitwise_not(src)           # 按位非
dst4 = cv.bitwise_xor(src1, src2)    # 按位异或
```

### 3\. 代码中的实验解读

```python
b1 = np.zeros((400, 400, 3), dtype=np.uint8)
b1[:, :] = (255, 0, 255)       # 品红色：B=255, G=0, R=255
b2 = np.zeros((400, 400, 3), dtype=np.uint8)
b2[:, :] = (0, 255, 255)       # 青黄色：B=0, G=255, R=255
```

#### 逐通道位运算过程

|通道|b1 像素值|b2 像素值|bitwise\_and|bitwise\_or|含义|
|---|---|---|---|---|---|
|B|255 → 11111111|0 → 00000000|0 → 00000000|255 → 11111111|AND：两个都是 1 才为 1|
|G|0 → 00000000|255 → 11111111|0 → 00000000|255 → 11111111|OR：有一个是 1 就为 1|
|R|255 → 11111111|255 → 11111111|255 → 11111111|255 → 11111111|全 255 则 AND/OR 都不变|

#### 结果图像的颜色

```Plain Text
b1 (品红):        (255, 0, 255)    紫色
b2 (青黄):        (0, 255, 255)    青色
bitwise_and:      (0, 0, 255)      红色      ← 只有 R 通道两个都是 255
bitwise_or:       (255, 255, 255)  白色      ← B/G/R 至少有一个是 255
```

🎨 这个实验直观展示了位运算的逐通道行为，用纯色图像做实验比复杂图片更容易理解底层原理。

### 4\. 八个二进制位具体怎么算

以 B 通道为例，255 vs 0：

```Plain Text
11111111  (255)
& 00000000  (0)
────────────
  00000000  (0)    ← AND 结果
  11111111  (255)
| 00000000  (0)
────────────
  11111111  (255)  ← OR 结果
```

### 5\. 位运算 vs 算术运算 对比

|维度|算术运算|位运算|
|---|---|---|
|操作层次|整数值（0\~255）|二进制位（每 bit 独立）|
|常用 API|cv\.add\(\) cv\.subtract\(\)|cv\.bitwise\_and\(\) cv\.bitwise\_or\(\)|
|典型场景|亮度调整、混合|掩码提取、抠图、ROI 操作|
|结果本质|数学加减乘除|二进制 AND/OR/NOT/XOR|

## 三、代码结构速查

### color\_table\_demo\(\) 函数

```Plain Text
① 定义 19 种 ColorMap 的列表
② 读取图像 → 显示原图
③ index 从 0 开始
④ while 循环：
   ├── applyColorMap(image, colormap[index % 19])
   ├── index += 1
   ├── imshow (每 500ms 自动刷新)
   └── 按 ESC 退出
⑤ 销毁窗口
```

### bitwise\_demo\(\) 函数

```Plain Text
① 创建两张 400×400 的纯色图像
② b1 = 品红色 (255,0,255)
③ b2 = 青色 (0,255,255)
④ 显示两张原图
⑤ bitwise_and → 红色 (0,0,255)
⑥ bitwise_or  → 白色 (255,255,255)
⑦ 按键后退出
```

## 四、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|cv\.applyColorMap\(\)|应用颜色表，伪彩色映射|Day 5 ✅|
|cv\.COLORMAP\_JET 等 19 种|内置颜色表|Day 5 ✅|
|cv\.bitwise\_and\(\)|按位与|Day 5 ✅|
|cv\.bitwise\_or\(\)|按位或|Day 5 ✅|
|bitwise\_xor\(\)|按位异或（未演示但同系列）|Day 5 ✅|
|index % len(list\)|取模循环技巧|Day 5 ✅|

## 五、明天计划

图像直方图：cv\.calcHist\(\)

直方图均衡化：cv\.equalizeHist\(\)

直方图应用：对比度提升
