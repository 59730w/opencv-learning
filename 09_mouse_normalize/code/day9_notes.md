# Day9：鼠标操作与响应 \+ 图像像素归一化

## 一、鼠标操作与响应 — 交互式矩形绘制

### 1\. 核心功能

鼠标按下左键 → 拖动 → 松开，在图像上**实时绘制矩形框**。松开后矩形消失，可重新绘制。

🌲 **林业应用场景**：手动标注树冠区域、框选病虫害叶片、交互式选择 ROI 研究区。

### 2\. `cv.setMouseCallback()` — 注册鼠标回调

```python
cv.setMouseCallback("mouse_demo", mouse_drawing)
```

|参数|含义|
|---|---|
|"mouse\_demo"|绑定到的窗口名称|
|mouse\_drawing|鼠标事件发生时调用的回调函数|

**核心机制**：当鼠标在 "mouse\_demo" 窗口上做任何操作（移动、点击、拖拽），OpenCV 自动调用 mouse\_drawing 函数。

### 3\. 回调函数签名

```python
def mouse_drawing(event, x, y, flags, param):
```

|参数|类型|含义|
|---|---|---|
|event|int|鼠标事件类型（点击、移动、松开等）|
|x|int|鼠标当前所在列坐标（图像 x 坐标）|
|y|int|鼠标当前所在行坐标（图像 y 坐标）|
|flags|int|附加状态（如是否同时按了 Ctrl/Shift）|
|param|any|setMouseCallback 传入的自定义参数（通常不用）|

### 4\. 三种鼠标事件

```python
cv.EVENT_LBUTTONDOWN    # 左键按下
cv.EVENT_MOUSEMOVE      # 鼠标移动
cv.EVENT_LBUTTONUP      # 左键松开
```

|事件|触发时机|在代码中做什么|
|---|---|---|
|EVENT\_LBUTTONDOWN|按下鼠标左键的瞬间|记录起点 (x1, y1\)|
|EVENT\_MOUSEMOVE|鼠标在窗口上移动|实时更新终点并绘制预览矩形|
|EVENT\_LBUTTONUP|松开鼠标左键的瞬间|绘制最终矩形，然后重置变量|

### 5\. 代码逐段解析

#### 阶段一：全局变量准备

```python
b1 = cv.imread("D:/images/test1.png")   # 显示用的图（会被修改）
img = np.copy(b1)                         # 原始图像的备份（用于恢复）
x1 = -1; x2 = -1; y1 = -1; y2 = -1       # 起点和终点坐标，-1 表示未设定
```

|变量|作用|
|---|---|
|b1|当前显示图像，会被反复修改（画矩形）|
|img|原始图像的纯净备份，用于擦除矩形|
|x1, y1|矩形左上角坐标（鼠标按下位置）|
|x2, y2|矩形右下角坐标（鼠标当前位置）|
|\-1|标记"没有有效的起点"，防止误触发|

#### 阶段二：左键按下 → 记录起点

```python
if event == cv.EVENT_LBUTTONDOWN:
    x1 = x
    y1 = y
```

x, y 由 OpenCV 自动传入，即鼠标按下时的坐标

此时只记录，不绘制

#### 阶段三：鼠标移动 → 实时预览

```python
if event == cv.EVENT_MOUSEMOVE:
    if x1 < 0 or y1 < 0:     # 还没按下左键？直接返回
        return
    x2 = x
    y2 = y
    dx = x2 - x1              # 计算宽度
    dy = y2 - y1              # 计算高度
    if dx > 0 and dy > 0:     # 确保是有效的矩形（右下方向拖动）
        b1[:, :, :] = img[:, :, :]                        # ① 先恢复原图（擦除旧矩形）
        cv.rectangle(b1, (x1,y1), (x2,y2), (0,0,255), 2)  # ② 绘制新矩形
```

##### 为什么先恢复再绘制？

```Plain Text
移动前：图上有旧矩形
  ↓ b1 = img（恢复原图）
清除旧矩形
  ↓ cv.rectangle(...)
画新矩形（位置随鼠标更新）
  ↓ imshow
显示 → 看起来像矩形跟随鼠标拖动
```

这个"恢复→重绘"模式是实现实时拖拽预览的标准做法，类似双缓冲机制。

##### 为什么要求 dx \> 0 且 dy \> 0？

确保矩形是从左上向右下拖动：

```Plain Text
✅ dx>0, dy>0:  起点左上 → 终点右下  ← 正常矩形
❌ dx<0:         向左拖动，OpenCV 不支持
❌ dy<0:         向上拖动，OpenCV 不支持
```

⚠️ 这只是一个简化实现。更完善的代码会用 min(x1,x2\), max(x1,x2\) 自动处理任意方向的拖动。

#### 阶段四：左键松开 → 定格 \+ 重置

```python
if event == cv.EVENT_LBUTTONUP:
    x2 = x; y2 = y
    dx = x2 - x1; dy = y2 - y1
    if dx > 0 and dy > 0:
        b1[:, :, :] = img[:, :, :]
        cv.rectangle(b1, (x1,y1), (x2,y2), (0,0,255), 2)
    x1 = -1; x2 = -1; y1 = -1; y2 = -1   # 重置，准备下一次绘制
```

松开时画最后一次（和移动时逻辑相同）

重置所有变量为 \-1，下次按下时重新记录新起点

#### 阶段五：主循环 \+ 退出

```python
def mouse_demo():
    cv.namedWindow("mouse_demo", cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("mouse_demo", mouse_drawing)
    while True:
        cv.imshow("mouse_demo", b1)    # 持续刷新显示
        c = cv.waitKey(10)             # 10ms 一帧
        if c == 27: break
    cv.destroyAllWindows()
```

|要素|说明|
|---|---|
|namedWindow|必须和 setMouseCallback 使用同一个窗口名|
|imshow(b1\)|每次循环显示最新状态的 b1|
|waitKey(10\)|维持循环，同时等待 ESC|

### 6\. 鼠标交互的完整流程图

```Plain Text
用户按下左键
  → EVENT_LBUTTONDOWN → 记录 (x1, y1)
  ↓
用户拖动鼠标
  → EVENT_MOUSEMOVE（连续触发）
  → 擦除旧矩形 → 画新矩形 → imshow 刷新
  ↓ （看起来矩形跟随鼠标缩放）
用户松开左键
  → EVENT_LBUTTONUP
  → 最终矩形定格 → 重置变量
  ↓
用户再次按下左键
  → 重新开始……
```

### 7\. 变量生命周期总结

|状态|x1, y1|x2, y2|矩形是否绘制|
|---|---|---|---|
|初始|\-1, \-1|\-1, \-1|❌ 不绘制|
|左键按下|有效坐标|\-1, \-1|❌ 还没有|
|拖动中|有效坐标|有效坐标|✅ 持续绘制|
|左键松开后|重置为 \-1|重置为 \-1|❌ 矩形留在图上，但不再更新|

## 二、图像像素归一化

### 1\. 什么是归一化，为什么需要

图像原始像素值范围：0 \~ 255（uint8 类型）。

|问题|说明|
|---|---|
|数值范围太大|深度学习模型对 0\~255 敏感，训练不稳定|
|不同类型的图值域不同|uint8 vs float32 无法直接混合运算|
|不同图像的亮度差异|同一算法对不同图片效果不一致|

**归一化解决方案**：把像素值映射到统一范围（通常是 0\.0 \~ 1\.0），同时转为浮点类型。

🌲 **林业应用**：深度学习模型（树种识别、植被分割）的输入必须归一化到 [0,1\] 或 [\-1,1\]。

### 2\. `cv.normalize()` — 核心 API

```python
cv.normalize(src, dst, alpha, beta, norm_type, dtype)
```

#### 参数详解

```python
cv.normalize(np.float32(image), result, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
```

|参数|值|含义|
|---|---|---|
|src|np\.float32(image\)|输入图像（必须先转为浮点类型）|
|dst|result|输出图像（预先创建的同尺寸零数组）|
|alpha|0|归一化范围下限（目标最小值）|
|beta|1|归一化范围上限（目标最大值）|
|norm\_type|cv\.NORM\_MINMAX|归一化类型|
|dtype|cv\.CV\_32F|输出数据类型（32位浮点）|

#### 归一化公式（NORM\_MINMAX）

```Plain Text
result = (src - min(src)) / (max(src) - min(src)) × (beta - alpha) + alpha
```

当 alpha=0, beta=1：

```Plain Text
result = (src - min) / (max - min)
       = 原像素值被线性映射到 [0, 1] 区间
```

|原像素值|归一化后|
|---|---|
|原图最小值（如 30）|→ 0\.0|
|原图中等值（如 142）|→ 约 0\.5|
|原图最大值（如 230）|→ 1\.0|

### 3\. 整个流程拆解

```python
# ① 读取原图（uint8, 0~255）
image = cv.imread("D:/images/test5.png")
# ② 转为 float32（必须！）
src_float = np.float32(image)              # 值还是 0~255，但类型是 float32
# ③ 创建接收结果的零数组（同尺寸、float32 类型）
result = np.zeros_like(src_float)          # shape 相同，全零，float32
# ④ 执行归一化：src → result，映射到 [0, 1]
cv.normalize(src_float, result, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
# ⑤ 显示——但 imshow 只能显示 uint8（0~255），所以要乘回去
cv.imshow("norm_demo", np.uint8(result * 255))
```

#### 为什么 imshow 前要 result \* 255？

```Plain Text
归一化后 result:  0.0 ~ 1.0 （float32）
imshow 期望:      0 ~ 255   （uint8）
result * 255:     0.0 ~ 255.0
np.uint8(...):    0 ~ 255     ← imshow 能正常显示
```

⚠️ 如果直接把 result（0\~1 浮点）传给 imshow，图片会全黑，因为 imshow 把 0\~1 当成 0\~255 解释。

### 4\. 为什么必须先转 float32

```python
# ❌ 错误做法
image = cv.imread("...")      # uint8
cv.normalize(image, ...)      # 试图归一化 uint8 → 结果也是 uint8（小数被截断）

# ✅ 正确做法
src = np.float32(image)       # 先转浮点
cv.normalize(src, result, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
```

uint8 只能存整数（0\~255），除以 255 后的小数部分全部被截断为 0。必须先转浮点类型，归一化才有意义。

### 5\. 常用的归一化类型

|归一化类型|公式|典型用途|
|---|---|---|
|cv\.NORM\_MINMAX|线性映射到 [alpha, beta\]|最常用，深度学习输入归一化|
|cv\.NORM\_L2|向量 L2 归一化（模长为 1）|特征匹配、相似度计算|
|cv\.NORM\_L1|向量 L1 归一化|稀疏特征|
|cv\.NORM\_INF|除以最大值|快速归一化|

### 6\. 归一化 vs 直接除以 255

```python
# 方式一：手动除以 255
result = np.float32(image) / 255.0     # 映射到 [0, 1]，但前提是 max=255

# 方式二：NORM_MINMAX
cv.normalize(src, result, 0, 1, cv.NORM_MINMAX)  # 自适应 min/max
```

|方式|效果|适用场景|
|---|---|---|
|/255\.0|固定映射：0→0, 255→1|标准 8bit 图像|
|NORM\_MINMAX|自适应：原图 min→0, 原图 max→1|值域不固定的图像（如深度图、热力图）|

普通 RGB 图像用 /255\.0 就够了。NORM\_MINMAX 的强大之处在于它的自适应性——无论输入范围是多少，都能映射到 [0, 1\]。

## 三、两个函数对比

|维度|mouse\_demo\(\)|norm\_demo\(\)|
|---|---|---|
|核心功能|鼠标拖拽画矩形|图像像素归一化|
|交互方式|鼠标事件驱动|无交互|
|核心 API|cv\.setMouseCallback\(\)|cv\.normalize\(\)|
|刷新方式|waitKey(10\) 循环|waitKey(0\) 一次性|
|数据类型|uint8|float32 ↔ uint8|
|应用方向|图像标注工具|深度学习数据预处理|

## 四、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|cv\.setMouseCallback\(\)|注册鼠标回调函数|Day 9 ✅|
|cv\.EVENT\_LBUTTONDOWN|鼠标左键按下事件|Day 9 ✅|
|cv\.EVENT\_MOUSEMOVE|鼠标移动事件|Day 9 ✅|
|cv\.EVENT\_LBUTTONUP|鼠标左键松开事件|Day 9 ✅|
|cv\.normalize\(\)|像素值归一化|Day 9 ✅|
|cv\.NORM\_MINMAX|线性归一化类型|Day 9 ✅|
|np\.float32\(\)|数组转为 32 位浮点|Day 9 ✅|
|cv\.CV\_32F|OpenCV 32位浮点数据类型|Day 9 ✅|
|np\.copy\(\)|图像深拷贝（用于恢复）|Day 9 ✅|

## 五、明天计划

图像放缩与插值

图像翻转
