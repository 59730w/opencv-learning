# Day8：随机数与随机颜色 \+ 多边形填充与绘制

## 一、随机数与随机颜色 — 动态线条艺术

### 1\. 核心思路

在黑色画布上，每 10ms 生成一条**随机起点、随机终点、随机颜色**的直线，形成动态变化的线条艺术效果。按 ESC 停止。

### 2\. `np.random.randint()` — 随机整数生成

```python
np.random.randint(low, high, size, dtype)
```

|参数|含义|代码示例|返回值|
|---|---|---|---|
|low|最小值（包含）|0|\-|
|high|最大值（不包含）|512|\-|
|size|生成几个数|2|长度为2的数组|
|dtype|数据类型|np\.int32|\-|

#### 代码中的三次调用

```python
xx = np.random.randint(0, 512, 2, dtype=np.int32)   # 两个随机数 → 起点x和终点x
yy = np.random.randint(0, 512, 2, dtype=np.int32)   # 两个随机数 → 起点y和终点y
bgr = np.random.randint(0, 255, 3, dtype=np.uint8)  # 三个随机数 → B, G, R
```

|变量|生成内容|用途|
|---|---|---|
|xx[0\], xx[1\]|0\~511 的两个随机整数|线段两个端点的 x 坐标（列）|
|yy[0\], yy[1\]|0\~511 的两个随机整数|线段两个端点的 y 坐标（行）|
|bgr[0\], bgr[1\], bgr[2\]|0\~254 的三个随机整数|线段的 B、G、R 颜色|

### 3\. cv\.line\(\) — 绘制直线

```python
cv.line(b1, (xx[0], yy[0]), (xx[1], yy[1]),
        (int(bgr[0]), int(bgr[1]), int(bgr[2])), 1, 8, 0)
```

|参数|值|含义|
|---|---|---|
|img|b1|在画布上绘制|
|pt1|(xx[0\], yy[0\]\)|起点坐标 (x, y\)|
|pt2|(xx[1\], yy[1\]\)|终点坐标 (x, y\)|
|color|(bgr[0\], bgr[1\], bgr[2\]\)|随机 BGR 颜色|
|thickness|1|线宽 1px|
|lineType|8|8连通线|
|shift|0|小数点不偏移|

#### ⚠️ 为什么需要 int\(\)

```python
(int(bgr[0]), int(bgr[1]), int(bgr[2]))
```

np\.random\.randint 返回的是 numpy\.int32 类型，OpenCV 的 C\+\+ 底层期望 Python 原生 int，直接传 numpy 类型可能报错或警告。显式转换最安全。

### 4\. waitKey\(10\) — 动态刷新

```python
c = cv.waitKey(10)    # 等待 10ms
```

|waitKey 值|效果|
|---|---|
|0|无限等待，直到按键|
|10|等 10ms，无按键也继续 → 动画效果|
|1|等 1ms → 最快刷新率|

10ms = 每秒约 100 条线，流畅的动态线条生成效果。

### 5\. 动态循环的标准模式

```python
canvas = np.zeros((512, 512, 3), dtype=np.uint8)   # 黑色画布
while True:
    # 每轮生成随机参数
    # 绘制 → 显示
    c = cv.waitKey(10)   # 10ms 刷新
    if c == 27:          # ESC 退出
        break
cv.destroyAllWindows()
```

这个模式通用性很强：后续做摄像头实时处理、视频逐帧处理，核心结构一模一样。

## 二、多边形填充与绘制

### 1\. 应用场景

|场景|说明|
|---|---|
|目标分割可视化|用多边形框出树冠、叶片等不规则形状|
|ROI 区域标注|标注不规则研究区域（非矩形）|
|语义分割展示|不同类别用不同颜色填充|
|交互式标注工具|鼠标点击生成多边形顶点|

🌲 **林业场景**：用多边形勾勒单株树冠轮廓、标记病虫害感染区域、圈画样方边界。

### 2\. 定义顶点坐标

```python
canvas = np.zeros((512, 512, 3), dtype=np.uint8)    # 512×512 黑色画布
pts = np.array([[100,100], [350,100], [450,280], [320,450], [80,400]],
               dtype=np.int32)
```

|顶点|坐标|位置|
|---|---|---|
|P0|(100, 100\)|左上|
|P1|(350, 100\)|右上|
|P2|(450, 280\)|右中|
|P3|(320, 450\)|右下|
|P4|(80, 400\)|左下|

```Plain Text
(100,100)————(350,100)
    |              |
    |            (450,280)
    |               |
 (80,400)—————(320,450)
```

顶点坐标格式：(x, y\)

OpenCV 会按顺序连接顶点，构成多边形

### 3\. 三种多边形绘制 API 对比

#### 方法一：cv\.fillPoly\(\) — 仅填充

```python
cv.fillPoly(canvas, [pts], (255, 0, 255), 8, 0)
```

[pts\]：列表的列表，每个内列表是一个多边形

(255, 0, 255\)：填充颜色（品红色）

只填充内部，不绘制边界

#### 方法二：cv\.polylines\(\) — 仅绘制轮廓

```python
cv.polylines(canvas, [pts], True, (0, 0, 255), 2, 8, 0)
```

|参数|值|含义|
|---|---|---|
|pts|[pts\]|多边形顶点|
|isClosed|True|是否闭合，True = 首尾相连|
|color|(0,0,255\)|红色边界|
|thickness|2|边界线宽 2px|

isClosed = False：不连接首尾，就是一条折线

只绘制边界，不填充内部

#### 方法三：cv\.drawContours\(\) — 一次搞定（推荐！）

```python
cv.drawContours(canvas, [pts], -1, (255, 0, 0), -1)
```

|参数|值|含义|
|---|---|---|
|image|canvas|目标画布|
|contours|[pts\]|轮廓列表|
|contourIdx|\-1|绘制所有轮廓（\-1 = 全部）|
|color|(255, 0, 0\)|蓝色|
|thickness|\-1|填充内部|

### 4\. thickness 参数的秘密

```python
cv.drawContours(canvas, [pts], -1, (255, 0, 0), -1)   # -1 = 填充
cv.drawContours(canvas, [pts], -1, (255, 0, 0), 1)    # 1 = 绘制1px宽的轮廓
cv.drawContours(canvas, [pts], -1, (255, 0, 0), 3)    # 3 = 绘制3px宽的轮廓
```

|thickness 值|效果|
|---|---|
|\-1|填充整个多边形内部|
|1, 2, 3\.\.\.|绘制轮廓线，数值 = 线宽（像素）|

一句话总结 drawContours：thickness 填 \-1 就是 fillPoly，填正数就是 polylines。一个 API 顶两个。

### 5\. 为什么顶点要包在 [pts\] 里

```python
cv.drawContours(canvas, [pts], -1, color, -1)
#                         ↑   ↑
#                       列表  每个元素是一个轮廓
```

drawContours 的第二个参数是轮廓列表：

```python
contours = [轮廓1, 轮廓2, 轮廓3, ...]
```

每个轮廓又是一个 (N, 2\) 的 NumPy 数组。所以即使只有 1 个多边形，也要写成 [pts\]。

#### 同时绘制多个多边形

```python
pts1 = np.array([[100,100], [200,100], [200,200], [100,200]])
pts2 = np.array([[300,300], [400,300], [400,400], [300,400]])
cv.drawContours(canvas, [pts1, pts2], -1, (0,255,0), -1)   # 两个都填充
```

contourIdx = \-1 表示所有轮廓都画。如果只想画第 0 个轮廓，改成 0。

### 6\. fillPoly / polylines / drawContours 终极对比

|API|能填充|能画轮廓|能一次画多个|推荐指数|
|---|---|---|---|---|
|cv\.fillPoly\(\)|✅|❌|✅|⭐⭐（功能单一）|
|cv\.polylines\(\)|❌|✅|✅|⭐⭐（功能单一）|
|cv\.drawContours\(\)|✅ thickness=\-1|✅ thickness\>0|✅|⭐⭐⭐⭐⭐|

**结论**：全部用 drawContours 就对了，一条 API 解决所有多边形需求。

## 三、两个函数对比

|维度|random\_color\_demo\(\)|polyline\_drawing\_demo\(\)|
|---|---|---|
|绘制内容|随机直线（动态）|固定多边形 \+ 填充|
|随机性|起点/终点/颜色全随机|无|
|更新方式|waitKey(10\) 循环刷新|一次性绘制后 waitKey(0\) 静止|
|核心 API|cv\.line\(\) \+ np\.random\.randint\(\)|cv\.drawContours\(\)|
|艺术效果|动态线条画|静态几何图形|

## 四、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|np\.random\.randint(low, high, size\)|生成随机整数数组|Day 8 ✅|
|cv\.line\(\)|绘制直线|Day 8 ✅|
|cv\.fillPoly\(\)|填充多边形（内部）|Day 8 ✅|
|cv\.polylines\(\)|绘制多边形轮廓|Day 8 ✅|
|cv\.drawContours\(\)|集大成者：填或画多边形|Day 8 ✅|
|thickness=\-1 用法|填充 vs 画轮廓的切换|Day 8 ✅|
|waitKey(10\) 动态刷新|10ms 间隔的动画循环|Day 8 ✅|

## 五、明天计划

鼠标操作与响应

图像像素类型转换与归一化
