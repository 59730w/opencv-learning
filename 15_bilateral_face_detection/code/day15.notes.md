# Day15：高斯双边模糊 \+ 实时人脸检测（DNN模块）

🎯 今天是学习 贾志刚OpenCV-Python 30 讲快速入门课程的**最后一天**。
从 Day1 的 `cv.imread()` 到今天加载深度学习模型做实时人脸检测——学习了 60\+ 个 API，更重要的是学会了 OpenCV 处理图像的完整思维框架。

## 一、高斯双边模糊 — `cv.bilateralFilter()`

### 1\. 回顾模糊滤波的进化路线

|Day|滤波方法|核心思想|致命缺陷|
|---|---|---|---|
|Day 14|`cv.blur()` 均值|邻域取平均|边缘和噪点一起模糊|
|Day 14|`cv.GaussianBlur()` 高斯|邻域加权平均（中心权重大）|边缘仍然会被模糊|
|**Day 15**|**`cv.bilateralFilter()`**** 双边**|**考虑空间距离 \+ 像素值差异**|**计算量大，速度慢**|

### 2\. 双边滤波的核心创新：两个高斯函数

普通高斯只考虑一个维度——**空间距离**（离中心像素越远，权重越小）。

双边滤波同时考虑**两个维度**：

**双边权重 = 空间权重 × 像素值权重**

空间权重：离中心像素越远 → 权重越小（和高斯一样）

像素值权重：像素值差异越大 → 权重越小（这是创新！）

|滤波类型|考虑的维度|效果|
|---|---|---|
|高斯|只有空间距离|边缘处的像素（空间近但颜色不同）也被平均 → **边缘模糊**|
|**双边**|**空间距离 \+ 像素值差异**|边缘处的像素（空间近但颜色差异大）权重极低 → **边缘保留**|

### 3\. 直观理解

图像中有一条边缘（左边暗=50，右边亮=200）：

暗区       边缘       亮区
(50\)       (50\|200\)    (200\)

**高斯滤波：**
中心像素在边缘上 → 邻域暗像素和亮像素都被平均 → 边缘被"抹平"了 → 50和200变成了125 → 模糊

**双边滤波：**
中心像素在边缘上 → 暗像素(50\)和中心(200\)差异大 → 权重极低 → 暗像素几乎不参与计算 → 中心保持200附近 → 边缘清晰！

🌲 **林业场景**：叶片图像去噪——既要去除传感器噪点，又要保留叶脉纹理和叶片边缘。双边滤波是**唯一能做到这件事的滤波器**。

### 4\. API 与参数

```python
result = cv.bilateralFilter(image, 0, 100, 10)
```

|参数|值|含义|
|---|---|---|
|src|image|输入图像|
|d|0|邻域直径（0 = 从 sigmaSpace 自动计算）|
|sigmaColor|100|像素值差异的高斯标准差（颜色相似度门槛）|
|sigmaSpace|10|空间距离的高斯标准差（空间邻近度门槛）|

### 5\. 两个 sigma 参数的调参指南

**sigmaColor（颜色敏感度）**：

├── 小 (10\~30\)：只有颜色非常接近的像素才参与模糊
│   → 边缘保留极好，但去噪效果弱

├── 中 (50\~100\)：颜色差异在一定范围内的像素参与
│   → 平衡去噪和边缘保留（代码中的选择）

└── 大 (150\~200\)：颜色差异很大也参与 → 退化为高斯模糊
    → 去噪强但边缘模糊

**sigmaSpace（空间敏感度）**：

├── 小 (5\~10\)：只有很近的像素参与 → 局部平滑

└── 大 (50\~100\)：大范围像素参与 → 全局平滑效果

### 6\. 双边滤波的优缺点

|优点|缺点|
|---|---|
|✅ 去噪同时保留边缘（独一无二）|❌ 计算量大，速度慢（～10\-20倍于高斯）|
|✅ 效果自然，"美颜磨皮"感|❌ 不适合实时处理大图|
|✅ 不会产生"块状模糊"|❌ 参数多，调参需要经验|
|✅ 适合纹理图像（叶片、皮肤、织物）|❌ 可能出现"油画效果"（阶梯效应）|

### 7\. 三种滤波效果速查

|场景|均值|高斯|双边|
|---|---|---|---|
|去除轻微噪点|✅|✅|✅|
|去除椒盐噪声|❌|❌|❌（用中值滤波）|
|保留边缘|❌|❌|✅|
|实时处理大图|✅|✅|❌|
|美颜磨皮|❌|✅|✅ 最佳|
|边缘检测前预处理|❌|✅ 推荐|❌ 太慢且没必要|

## 二、OpenCV DNN 模块 — 深度学习模型推理

### 1\. 什么是 DNN 模块

OpenCV 的 `cv.dnn` 模块允许你加载预训练好的深度学习模型并进行推理，无需安装 PyTorch/TensorFlow。

PyTorch/TensorFlow 训练模型 → 导出为开放格式（\.pb / \.onnx / \.caffemodel）
↓
OpenCV DNN 模块加载 → 推理

|支持的框架|文件格式|
|---|---|
|TensorFlow|\.pb（模型）\+ \.pbtxt（配置）|
|Caffe|\.caffemodel \+ \.prototxt|
|ONNX|\.onnx|
|Darknet|\.cfg \+ \.weights|
|OpenVINO|\.xml \+ \.bin|

这意味着：OpenCV 不只是图像处理库，它还是一个轻量级深度学习推理引擎。

## 三、实时人脸检测 — 完整流程拆解

### 完整 Pipeline

① 加载预训练模型
↓
② 打开视频流
↓
③ while 循环：
　├── 读取一帧
　├── 预处理 (blobFromImage\)
　├── 模型推理 (setInput \+ forward\)
　├── 解析输出 (坐标 \+ 置信度\)
　├── 绘制边框 \+ 标签
　├── 计算 FPS
　└── 显示结果
↓
④ ESC 退出

### 步骤 ①：加载模型

```python
model_bin = "D:/pycharm_text1.0/opencv_1/opencv_face_detector_uint8.pb"
config_text = "D:/pycharm_text1.0/opencv_1/opencv_face_detector.pbtxt"
net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
```

|文件|作用|
|---|---|
|\.pb (protobuf\)|模型权重（训练好的神经网络参数）|
|\.pbtxt|模型结构描述（网络有哪些层、每层怎么连接）|
|cv\.dnn\.readNetFromTensorflow\(\)|加载 TensorFlow 格式的模型|

### 步骤 ②：打开视频

```python
capture = cv.VideoCapture("D:/TV sucai/01.mp4")
```

回顾 Day 11：打开视频文件，准备逐帧处理。

### 步骤 ③：逐帧循环

#### 3\.1 读取帧 \+ 获取尺寸

```python
ret, frame = capture.read()
if ret is not True:
    break
h, w, c = frame.shape
```

h, w, c：帧的高度、宽度、通道数

#### 3\.2 预处理：cv\.dnn\.blobFromImage\(\)（核心！）

```python
blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
```

|参数|值|含义|
|---|---|---|
|image|frame|输入帧（BGR 格式）|
|scalefactor|1\.0|像素值缩放因子（1\.0 = 不缩放）|
|size|(300, 300\)|模型要求输入尺寸 → 所有帧缩放到 300×300|
|mean|(104\.0, 177\.0, 123\.0\)|均值削减值（B, G, R），每个通道减去对应均值|
|swapRB|False|是否交换 R/B 通道（True = BGR→RGB）|
|crop|False|是否居中裁剪|

**均值削减的作用**

原始 BGR 像素：(150, 200, 180\)
减去 mean：    (104, 177, 123\)
结果：         (46, 23, 57\)

→ 数据中心化，让模型输入归一化到零附近
→ (104, 177, 123\) 是训练该模型的训练集均值
这个均值值是模型特定的——不同模型有不同的 mean 值，不能随便改。

**输出 blob 的形状**

blobImage\.shape → (1, 3, 300, 300\)
│  │   │    └── 宽度
│  │   └────── 高度
│  └────────── 通道数（RGB 三通道）
└───────────── batch size = 1（一次处理一张图）

OpenCV DNN 的 blob 是 NCHW 格式（批大小、通道、高、宽），和 PyTorch 默认格式一致。

#### 3\.3 模型推理

```python
net.setInput(blobImage)      # 把预处理好的 blob 送入网络输入层
cvOut = net.forward()        # 执行前向传播，获取输出
```

|步骤|操作|说明|
|---|---|---|
|setInput\(\)|设置输入|把 blob 送进网络的第一个层|
|forward\(\)|前向传播|数据流经所有层，输出检测结果|

#### 3\.4 解析输出

cvOut\.shape   → (1, 1, N, 7\)
│  │  │  └── 每个检测结果 7 个值
│  │  └───── 检测到 N 个目标
│  └──────── 1（预留维度）
└─────────── batch size = 1

**每个检测结果的 7 个值：**

|索引|含义|说明|
|---|---|---|
|\[0\]|批次索引|固定为 0|
|\[1\]|类别索引|0=背景, 1=人脸|
|\[2\]|置信度 score|0\~1，越大越可信|
|\[3\]|left（左边界）|归一化坐标（0\~1），需 × 原图宽度|
|\[4\]|top（上边界）|归一化坐标（0\~1），需 × 原图高度|
|\[5\]|right（右边界）|归一化坐标（0\~1），需 × 原图宽度|
|\[6\]|bottom（下边界）|归一化坐标（0\~1），需 × 原图高度|

#### 3\.5 遍历检测结果并绘制

```python
for detection in cvOut[0, 0, :, :]:
    score = float(detection[2])
    if score > 0.5:                                    # 置信度 > 0.5 才认为是有效检测
        left   = detection[3] * w                       # 归一化坐标 × 原图宽度
        top    = detection[4] * h
        right  = detection[5] * w
        bottom = detection[6] * h
        cv.rectangle(frame, (int(left), int(top)),
                     (int(right), int(bottom)), (255, 0, 0), 2)
        cv.putText(frame, "score:%.2f" % score,
                   (int(left), int(top)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
```

**关键技巧：归一化坐标 → 像素坐标**

模型输出的坐标是 0\~1 之间的归一化值（相对于 300×300 输入图）
乘以原图尺寸 → 映射到实际像素坐标
left × w:   如果 left=0\.3, w=1920 → 0\.3×1920 = 576 像素

**为什么阈值设为 0\.5**

score \> 0\.5：只显示置信度大于 50% 的检测框
score \> 0\.8：更严格，减少误检但可能漏检
score \> 0\.3：更宽松，增加召回但误检更多
0\.5 是目标检测的通用默认阈值，可根据实际需求调整。

### 步骤 ④：FPS 计算（性能监控）

```python
e1 = cv.getTickCount()                                    # 帧开始时的时间戳
# ... 所有处理逻辑 ...
e2 = cv.getTickCount()                                    # 帧结束时的时间戳
fps = cv.getTickFrequency() / (e2 - e1)                    # FPS = 时钟频率 / 时间差
```

|API|返回值|含义|
|---|---|---|
|cv\.getTickCount\(\)|整数|系统启动以来的时钟周期数|
|cv\.getTickFrequency\(\)|浮点（约 10^7\~10^9）|每秒的时钟周期数（取决于 CPU）|

处理一帧的耗时 = (e2 \- e1\) / cv\.getTickFrequency\(\)     （秒）
FPS = 1 / 耗时 = cv\.getTickFrequency\(\) / (e2 \- e1\)

### 步骤 ⑤：模型推理时间（性能分析）

```python
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
```

|API|返回值|含义|
|---|---|---|
|net\.getPerfProfile\(\)|\(总时间, 各层时间列表\)|网络推理的耗时（时钟周期数）|

推理时间 = t / cv\.getTickFrequency\(\)         （秒）
        = t \* 1000 / cv\.getTickFrequency\(\)   （毫秒）

|指标|计算方式|包含内容|
|---|---|---|
|FPS|getTickCount\(\) 前后差值|一整帧的所有操作（读取、预处理、推理、绘制、显示）|
|推理时间|net\.getPerfProfile\(\)|仅模型推理（forward），不含预处理和绘制|

FPS 低 \+ 推理时间低  → 瓶颈在预处理/绘制/显示
FPS 低 \+ 推理时间高  → 瓶颈在模型本身

### 步骤 ⑥：显示信息叠加

```python
cv.putText(frame, label + (" FPS: %.2f" % fps), (10, 50),
           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv.imshow('face-detection-demo', frame)
```

回顾 Day 7：cv\.putText\(\) 在图像上叠加文字信息。

### 步骤 ⑦：退出机制

```python
c = cv.waitKey(1)
if c == 27:       # ESC
    break
```

waitKey(1\)：每帧等 1ms → 最大帧率约 1000fps（实际受限于处理速度）
waitKey(0\) 会导致视频卡住等按键

## 四、实时人脸检测完整框架模板

```python
# ① 加载模型
net = cv.dnn.readNetFromTensorflow("model.pb", "config.pbtxt")
# ② 打开视频
cap = cv.VideoCapture("video.mp4")
while True:
    # ③ 读取一帧
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    # ④ 预处理
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104, 177, 123), False, False)
    # ⑤ 推理
    net.setInput(blob)
    detections = net.forward()
    # ⑥ 解析 + 绘制
    for det in detections[0, 0, :, :]:
        score = float(det[2])
        if score > 0.5:
            left   = int(det[3] * w)
            top    = int(det[4] * h)
            right  = int(det[5] * w)
            bottom = int(det[6] * h)
            cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv.putText(frame, f"{score:.2f}", (left, top - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # ⑦ 显示
    cv.imshow("result", frame)
    if cv.waitKey(1) == 27:
        break
cap.release()
cv.destroyAllWindows()
```

这个模板稍加修改，就能适配行人检测、车牌检测、手势识别等任何目标检测任务。

## 五、两个函数对比

|维度|bifilter\_demo\(\)|face\_detection\_demo\(\)|
|---|---|---|
|核心功能|去噪 \+ 保留边缘|加载 DNN 模型做实时人脸检测|
|核心 API|cv\.bilateralFilter\(\)|cv\.dnn\.readNetFromTensorflow\(\) \+ blobFromImage\(\) \+ forward\(\)|
|理论基础|双边滤波（空间\+像素值双高斯）|深度学习（SSD 目标检测网络）|
|输入|单张图片|视频流（逐帧）|
|输出|模糊后图片|带检测框和标签的视频帧|
|计算量|中等|高（涉及神经网络推理）|
|跨 Day 知识|Day 14（高斯滤波）|Day 11（视频读取）\+ Day 7（绘图）\+ Day 4（键盘交互）|

## 六、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|cv\.bilateralFilter\(\)|双边滤波（去噪保留边缘）|Day 15 ✅|
|cv\.dnn\.readNetFromTensorflow\(\)|加载 TensorFlow 模型|Day 15 ✅|
|cv\.dnn\.blobFromImage\(\)|图像预处理（缩放\+均值削减\+通道转换）|Day 15 ✅|
|net\.setInput\(\)|设置网络输入|Day 15 ✅|
|net\.forward\(\)|执行前向传播推理|Day 15 ✅|
|net\.getPerfProfile\(\)|获取推理耗时（性能分析）|Day 15 ✅|
|cv\.getTickCount\(\)|获取系统时钟周期数（计时用）|Day 15 ✅|
|cv\.getTickFrequency\(\)|获取时钟频率（秒/周期）|Day 15 ✅|
|sigmaColor / sigmaSpace|双边滤波的参数|Day 15 ✅|

## 七、30 讲课程回顾

|阶段|Day|核心内容|
|---|---|---|
|基础入门|1\-2|环境搭建、imread/imshow、色彩空间|
|像素操作|3\-4|像素读写、ROI、算术运算|
|交互与色彩|5\-6|TrackBar、逻辑运算、ColorMap|
|标注与绘制|7\-8|几何形状、鼠标交互、多边形|
|变换与视频|9\-10|归一化、缩放、翻转|
|旋转与视频处理|11\-12|仿射变换、VideoCapture、VideoWriter、直方图|
|直方图与滤波|13\-14|二维直方图、均衡化、卷积、高斯模糊|
|高级滤波与深度学习|15|双边滤波、DNN 模块、人脸检测|

