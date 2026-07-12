# Day12：视频处理与保存 \+ 图像直方图

## 开篇：matplotlib 中文配置

```python
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
```

|配置项|作用|
|---|---|
|plt\.switch\_backend('TkAgg'\)|切换到 Tkinter 后端，确保窗口能正常弹出显示|
|font\.family = 'SimHei'|使用黑体字体，解决 matplotlib 中文乱码问题|
|axes\.unicode\_minus = False|正常显示负号（不设置的话负号会变成方块）|

⚠️ 这三行配置必须在 plt\.plot\(\) 之前执行，否则不生效。

## 一、视频处理与保存 — `cv.VideoWriter()`

### 1\. 从 Day11 到 Day12 的升级

|Day 11|Day 12|
|---|---|
|读取视频 → 显示|读取视频 → 处理 → 保存为新视频|
|只看不存|处理结果落盘，可回放、可分析|

🌲 **林业应用场景**：读取无人机航拍视频 → 每帧做植被分割 → 保存标注后的视频 → 用于演示或后续分析。

### 2\. 完整流程

```Plain Text
打开源视频 → 获取属性 → 创建输出视频 → 逐帧循环
                                      ↓
                              读取 → 处理(HSV) → 显示 → 写入
                                      ↓
                              按ESC退出 → 释放所有资源
```

### 3\. `cv.VideoWriter()` — 创建视频写入器

```python
fourcc = cv.VideoWriter.fourcc(*'mp4v')
out = cv.VideoWriter("D:/images/test6.mp4", fourcc, fps, (int(w), int(h)), True)
```

#### 参数详解

|参数|值|含义|
|---|---|---|
|filename|"D:/images/test6\.mp4"|输出视频文件路径|
|fourcc|cv\.VideoWriter\.fourcc(\*'mp4v'\)|视频编码格式（四字符编码）|
|fps|cap\.get(cv\.CAP\_PROP\_FPS\)|输出视频的帧率（与原视频保持一致）|
|frameSize|(int(w\), int(h\)\)|输出视频每帧的尺寸（宽, 高）|
|isColor|True|True = 彩色视频，False = 灰度视频|

#### 关键细节

##### (1\) (int(w\), int(h\)\) — 为什么需要 int\(\)

```python
w = cap.get(cv.CAP_PROP_FRAME_WIDTH)    # 返回值是 float（如 1920.0）
h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)   # 返回值是 float（如 1080.0）
```

VideoWriter 要求尺寸参数必须是整数，所以显式用 int\(\) 转换。

##### (2\) isColor=True — 彩色还是灰度

|isColor|含义|帧格式要求|
|---|---|---|
|True|彩色视频|每帧必须是 3 通道 BGR|
|False|灰度视频|每帧必须是 1 通道灰度|

如果 isColor=True 但写入灰度帧 → 报错或输出异常。两者必须匹配。

### 4\. FourCC — 视频编码格式

```python
fourcc = cv.VideoWriter.fourcc(*'mp4v')
```

#### FourCC 是什么

FourCC = Four Character Code = 四个字符的视频编码标识。

#### \*'mp4v' 的语法解析：

```python
cv.VideoWriter.fourcc(*'mp4v')
# 等价于
cv.VideoWriter.fourcc('m', 'p', '4', 'v')
```

\* 把字符串 'mp4v' 拆成四个独立字符传入函数。

#### 常用 FourCC 编码

|FourCC|文件格式|质量|兼容性|
|---|---|---|---|
|\*'mp4v'|\.mp4|高|⭐⭐⭐ Windows/Mac 通用|
|\*'XVID'|\.avi|高|⭐⭐⭐ 通用|
|\*'MJPG'|\.avi|中（运动 JPEG）|⭐⭐|
|\*'H264'|\.mp4|最高（压缩率好）|⭐（需要额外安装）|

推荐默认用 `*'mp4v'`，生成 MP4 文件体积小、兼容性好。如果遇到无法打开，换 `*'XVID'`。

### 5\. out\.write\(\) — 写入一帧

```python
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)   # 处理：BGR → HSV
out.write(hsv)                                 # 写入 HSV 帧
```

|必须满足的条件|说明|
|---|---|
|帧尺寸 = (w, h\)|和 VideoWriter 创建时的尺寸必须一致|
|通道数匹配 isColor|isColor=True → 3 通道，isColor=False → 1 通道|
|数据类型 uint8|OpenCV 图像默认就是 uint8|

⚠️ 如果写入帧尺寸不一致 → 不报错但输出视频可能为空或只有部分帧。

### 6\. 资源释放

```python
out.release()    # 关闭视频写入器
cap.release()    # 关闭视频读取器
cv.destroyAllWindows()
```

|调用顺序|说明|
|---|---|
|先 out\.release\(\)|确保所有帧写入磁盘，关闭输出文件|
|再 cap\.release\(\)|释放视频文件句柄|
|最后 destroyAllWindows\(\)|关闭所有 GUI 窗口|

如果忘记 out\.release\(\)，输出视频文件可能损坏或大小为 0。这是高频踩坑点。

### 7\. 视频处理循环完整模板

```python
cap = cv.VideoCapture("input.mp4")
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter.fourcc(*'mp4v')
out = cv.VideoWriter("output.mp4", fourcc, fps, (w, h), True)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # ======== 在这里处理每一帧 ========
    processed = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # ================================
    cv.imshow("frame", frame)
    cv.imshow("result", processed)
    out.write(processed)                     # 写入输出视频
    if cv.waitKey(10) == 27:
        break
out.release()
cap.release()
cv.destroyAllWindows()
```

## 二、图像直方图 — `cv.calcHist()`

### 1\. 什么是直方图

直方图 = 统计每个像素值出现了多少次。

|概念|说明|
|---|---|
|X 轴|像素值（0 \~ 255）|
|Y 轴|该像素值在图像中出现的次数（频数）|
|意义|反映图像的亮度分布、对比度、曝光情况|

#### 为什么需要直方图

|应用|说明|
|---|---|
|曝光判断|直方图偏左 = 偏暗（欠曝），偏右 = 偏亮（过曝）|
|对比度评估|分布宽 = 对比度高，分布窄 = 图像发灰|
|阈值选取|双峰直方图 → 谷底就是最佳二值化阈值|
|颜色分析|三个通道直方图反映图像偏色情况|

🌲 **林业场景**：分析植被图像的绿色通道直方图，判断植被覆盖度和健康状态；NDVI 直方图统计植被指数分布。

### 2\. `cv.calcHist()` — 核心 API

```python
hist = cv.calcHist([image], [i], None, [256], [0, 256])
```

#### 参数详解

|参数|值|含义|
|---|---|---|
|images|[image\]|输入图像列表（必须用列表包起来）|
|channels|[i\]|要统计的通道索引（也必须用列表）|
|mask|None|掩码，None = 统计整张图|
|histSize|[256\]|分成多少组（bin 数量）|
|ranges|[0, 256\]|像素值范围（0\~255，上限不包含）|

#### 通道索引

```python
[0]   → B 通道（蓝色）
[1]   → G 通道（绿色）
[2]   → R 通道（红色）
```

#### 返回值 hist

形状 (256, 1\) 的 NumPy 数组

hist[0\] = 像素值为 0 的像素个数

hist[255\] = 像素值为 255 的像素个数

### 3\. 三通道直方图绘制

```python
color = ('blue', 'green', 'red')                      # BGR 对应的颜色
for i, color in enumerate(color):                      # i=0/1/2, color='blue'/'green'/'red'
    hist = cv.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)                        # 用对应颜色绘制折线
    plt.xlim([0, 256])                                 # X 轴范围 0~255
plt.show()
```

#### 遍历过程

|循环次数|i|color|统计内容|绘制颜色|
|---|---|---|---|---|
|第 1 次|0|'blue'|B 通道直方图|蓝色|
|第 2 次|1|'green'|G 通道直方图|绿色|
|第 3 次|2|'red'|R 通道直方图|红色|

#### enumerate\(\) 的作用

```python
for i, color in enumerate(('blue', 'green', 'red')):
# i=0, color='blue'
# i=1, color='green'
# i=2, color='red'
```

enumerate 同时给出索引和值，省去手动计数。

### 4\. plt\.xlim\(\) — 设置 X 轴范围

```python
plt.xlim([0, 256])
```

确保 X 轴从 0 到 255（覆盖完整的 256 个灰度级）

如果不设置，matplotlib 可能自动裁剪

### 5\. plt\.plot\(\) vs OpenCV 的 imshow\(\)

|特性|plt\.plot\(\)|cv\.imshow\(\)|
|---|---|---|
|用途|绘制统计图、曲线图|显示图像|
|显示内容|直方图曲线|像素颜色|
|坐标系|数值坐标|像素坐标|
|可交互|缩放、平移、保存|基本显示|

### 6\. 如何读懂直方图

```Plain Text
直方图偏左（大部分像素值 < 100）
→ 图像偏暗（欠曝）
直方图偏右（大部分像素值 > 150）
→ 图像偏亮（过曝）
直方图均匀分布（0~255 都有）
→ 对比度好，细节丰富
直方图聚集在中间（集中在 100~150）
→ 图像发灰，对比度低
直方图出现两个峰
→ 图像有前景/背景两个主色调（双峰图）
```

#### 三通道可视化

```Plain Text
蓝色曲线偏高     → 图像偏蓝
绿色曲线偏高     → 图像偏绿（植被图常见）
红色曲线偏高     → 图像偏红（黄昏/秋天场景）
三条曲线重合     → 灰度图或中性色调
```

### 7\. 直方图的局限

|局限性|说明|
|---|---|
|丢失空间信息|直方图只统计"有多少"，不记录"在哪里"|
|不同图像可能同直方图|完全打乱像素位置，直方图不变|
|不能判断内容|直方图不知道拍的是人脸还是森林|

## 三、两个函数对比

|维度|video\_save\_demo\(\)|image\_hist\(\)|
|---|---|---|
|核心功能|读取视频 → 逐帧处理 → 保存新视频|计算并绘制 BGR 三通道直方图|
|核心 API|VideoWriter \+ out\.write\(\)|cv\.calcHist\(\) \+ plt\.plot\(\)|
|处理方式|逐帧循环，实时处理|一次性统计整张图|
|输出形式|新视频文件（\.mp4）|matplotlib 图表窗口|
|资源释放|out\.release\(\) \+ cap\.release\(\)|—|
|跨库协作|纯 OpenCV|OpenCV \+ matplotlib|

## 四、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|cv\.VideoWriter\(\)|创建视频写入器|Day 12 ✅|
|cv\.VideoWriter\.fourcc(\*'mp4v'\)|指定视频编码格式|Day 12 ✅|
|out\.write(frame\)|写入一帧到视频|Day 12 ✅|
|out\.release\(\)|关闭视频写入器（完成保存）|Day 12 ✅|
|cap\.release\(\)|释放视频资源|Day 12 ✅|
|cv\.calcHist\(\)|计算图像直方图|Day 12 ✅|
|plt\.plot\(\)|matplotlib 绘制曲线|Day 12 ✅|
|plt\.xlim\(\)|设置 X 轴范围|Day 12 ✅|
|plt\.show\(\)|显示 matplotlib 图表|Day 12 ✅|
|enumerate\(\)|同时遍历索引和值|Day 12 ✅|
|plt\.switch\_backend('TkAgg'\)|设置 matplotlib 后端|Day 12 ✅|
|rcParams 中文字体配置|解决 matplotlib 乱码|Day 12 ✅|

## 五、明天计划

二维直方图

直方图均衡化
