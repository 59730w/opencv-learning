# Day11：图像旋转 \+ 视频文件/摄像头使用

## 一、图像旋转 — `cv.getRotationMatrix2D()` \+ `cv.warpAffine()`

### 1\. 为什么旋转比翻转复杂

Day 10 的 `cv.flip()` 只做了像素索引重排，不改变图像尺寸。

旋转任意角度则完全不同：

|问题|说明|
|---|---|
|图像会超出原画布|旋转后四个角的位置变了，原画布装不下|
|新画布尺寸未知|需要根据旋转角度计算外接矩形大小|
|图像可能被裁切|如果还用原尺寸，旋转后的内容可能被截掉|

\> **解决方案**：先算新画布尺寸 → 调整平移量让图像居中 → 在新画布上做旋转。

### 2\. 完整流程拆解（四步走）

原始图像 → 计算新尺寸 → 获取旋转矩阵 → 调整平移 → 仿射变换

800×600 → 45°旋转后 → 绕中心旋转 → 居中到新画布 → 最终结果

new\_w × new\_h 、 M[0,2\] \+= Δx 、 M[1,2\] \+= Δy 、 完整显示

### 3\. 第一步：计算旋转后的新画布尺寸

```python
angle = 45                         # 旋转角度
rad = np.radians(angle)            # 角度 → 弧度
cos_abs = abs(np.cos(rad))         # |cosθ|
sin_abs = abs(np.sin(rad))         # |sinθ|
new_w = int(w * cos_abs + h * sin_abs)   # 新宽度
new_h = int(w * sin_abs + h * cos_abs)   # 新高度
```

#### 为什么取绝对值？

旋转角度可能 \> 90°，cos/sin 可能为负数。外接矩形只关心边长，不关心方向，所以取绝对值。

#### 公式推导（直观理解）

```Plain Text
┌──────────┐  w          旋转 45° 后，原来的宽 w 在新坐标系中：
│   原图   │              - 水平方向投影：w × cosθ
│  w × h  │ h            - 垂直方向投影：w × sinθ
└──────────┘
                         同理，原来的高 h：
                         - 水平方向投影：h × sinθ
                         - 垂直方向投影：h × cosθ
新宽度 = w 的水平分量 + h 的水平分量 = w·cosθ + h·sinθ
新高度 = w 的垂直分量 + h 的垂直分量 = w·sinθ + h·cosθ
```

|原图尺寸|角度|新尺寸（约）|
|---|---|---|
|800×600|0°|800×600（不变）|
|800×600|45°|990×990|
|800×600|90°|600×800（宽高互换）|
|800×600|180°|800×600（不变）|

### 4\. np\.radians\(\) — 角度转弧度

```python
rad = np.radians(45)    # 45° → 0.785398163... 弧度
```

|函数|方向|示例|
|---|---|---|
|np\.radians(deg\)|角度 → 弧度|np\.radians(180\) → 3\.1415926 (π\)|
|np\.degrees(rad\)|弧度 → 角度|np\.degrees(np\.pi\) → 180°|

NumPy 的三角函数（sin、cos、tan）全部使用弧度，所以必须先转换。

### 5\. 第二步：获取旋转矩阵

```python
M = cv.getRotationMatrix2D((w/2, h/2), angle, 1.0)
```

#### 函数签名

```python
M = cv.getRotationMatrix2D(center, angle, scale)
```

|参数|值|含义|
|---|---|---|
|center|(w/2, h/2\)|旋转中心（图像正中心）|
|angle|45|旋转角度（角度制，正=逆时针）|
|scale|1\.0|缩放因子（1\.0 = 不缩放，\>1 放大，\<1 缩小）|

#### 返回值 M

一个 2×3 的仿射变换矩阵：

```Plain Text
M = |  cosθ   sinθ   tx  |
    | -sinθ   cosθ   ty  |
```

左上 2×2：旋转 \+ 缩放

右边一列（tx, ty）：平移

```Plain Text
M 矩阵中：
M[0, 0] = cosθ         M[0, 1] = sinθ         M[0, 2] = tx
M[1, 0] = -sinθ        M[1, 1] = cosθ         M[1, 2] = ty
```

### 6\. 第三步：调整平移量，使图像在新画布居中

```python
M[0, 2] += (new_w / 2 - w / 2)    # 水平方向平移量调整
M[1, 2] += (new_h / 2 - h / 2)    # 垂直方向平移量调整
```

#### 为什么要调整？

getRotationMatrix2D 默认绕原图中心旋转，但新画布比原图大，如果不调整平移量，旋转后的图像会偏到画布一角。

```Plain Text
新画布 (990×990)
┌─────────────────────┐
│                     │
│      ┌──────┐       │  ← 如果不调整，旋转后的图像可能在这里（偏左上方）
│      │ 原图 │       │
│      └──────┘       │
│                     │
└─────────────────────┘
调整后：
┌─────────────────────┐
│                     │
│    ┌──────────┐     │  ← 图像居中显示
│    │  旋转后  │     │
│    └──────────┘     │
│                     │
└─────────────────────┘
```

#### 平移量计算公式

```Plain Text
Δx = new_w/2 - w/2    -- 新画布中心 - 旧画布中心 = 需要额外平移的距离
Δy = new_h/2 - h/2
M[0, 2] += Δx         -- 把额外平移量加到 M 矩阵的 tx 分量上
M[1, 2] += Δy         -- 把额外平移量加到 M 矩阵的 ty 分量上
```

这是整个旋转操作最精妙的一步——手动修改仿射矩阵的平移分量，让旋转后的图像完完整整地显示在新画布正中央。

### 7\. 第四步：执行仿射变换

```python
dst = cv.warpAffine(src, M, (new_w, new_h))
```

#### 函数签名

```python
dst = cv.warpAffine(src, M, dsize)
```

|参数|值|含义|
|---|---|---|
|src|src|输入图像|
|M|2×3 矩阵|仿射变换矩阵|
|dsize|(new\_w, new\_h\)|输出图像尺寸（宽, 高）|

warpAffine 对每个像素执行：

```Plain Text
新坐标 (x', y') = M × 原坐标 (x, y, 1)^T
```

如果计算出的原坐标不在图像范围内（旋转后露出背景），则填充黑色（0, 0, 0）。

### 8\. 图像旋转完整流程回顾

```Plain Text
① 读图 → 获取尺寸 (h, w)
② 选角度 angle
③ np.radians(angle) → rad
④ 计算新画布尺寸：
   new_w = w·cosθ + h·sinθ
   new_h = w·sinθ + h·cosθ
⑤ cv.getRotationMatrix2D((w/2,h/2), angle, 1.0) → M
⑥ 调整平移量：
   M[0,2] += (new_w/2 - w/2)
   M[1,2] += (new_h/2 - h/2)
⑦ cv.warpAffine(src, M, (new_w, new_h)) → dst
⑧ 显示结果
```

这套流程是任意角度旋转且不裁切的标准方案，可以直接复制到项目中复用。

### 9\. 仿射变换能做什么

cv\.warpAffine\(\) 不只能旋转，它是所有线性变换的统一接口：

|操作|如何构造 M|API 快捷方式|
|---|---|---|
|平移|手动构造 M|—|
|旋转|cv\.getRotationMatrix2D\(\)|—|
|缩放|修改 M[0,0\] 和 M[1,1\]|cv\.resize\(\)|
|仿射（三点）|cv\.getAffineTransform\(\)|—|

## 二、视频文件 / 摄像头使用 — `cv.VideoCapture()`

### 1\. 视频的本质

视频 = 一系列连续图像帧按时间顺序播放。

每一帧就是一张 BGR 图像（和 cv\.imread\(\) 读到的一模一样）

FPS（Frames Per Second）= 每秒播放多少帧

所以处理视频 = 在一个循环里逐帧读取、逐帧处理

🌲 **林业场景**：读取无人机航拍视频、处理摄像头实时画面、视频中检测野生动物、分析植被时序变化。

### 2\. `cv.VideoCapture()` — 打开视频源

```python
cap = cv.VideoCapture("D:/TV sucai/01.mp4")    # 打开视频文件
# cap = cv.VideoCapture(0)                     # 打开摄像头（0=默认摄像头）
```

|参数|含义|
|---|---|
|"路径/01\.mp4"|视频文件路径|
|0|默认摄像头（第一个摄像头）|
|1, 2, \.\.\.|其他摄像头|
|RTSP/HTTP 地址|网络摄像头流|

### 3\. 获取视频属性

```python
w   = cap.get(cv.CAP_PROP_FRAME_WIDTH)     # 帧宽度（像素）
h   = cap.get(cv.CAP_PROP_FRAME_HEIGHT)    # 帧高度（像素）
fps = cap.get(cv.CAP_PROP_FPS)             # 帧率
```

#### 常用属性

|属性常量|含义|
|---|---|
|cv\.CAP\_PROP\_FRAME\_WIDTH|视频帧宽度|
|cv\.CAP\_PROP\_FRAME\_HEIGHT|视频帧高度|
|cv\.CAP\_PROP\_FPS|帧率（帧/秒）|
|cv\.CAP\_PROP\_FRAME\_COUNT|视频总帧数|
|cv\.CAP\_PROP\_POS\_FRAMES|当前读取到第几帧|

### 4\. 逐帧读取循环（核心模式）

```python
while True:
    ret, frame = cap.read()       # 读取一帧
    if ret is not True:           # ret=False → 视频结束
        break
    cv.imshow("frame", frame)     # 显示当前帧
    c = cv.waitKey(10)            # 等待 10ms
    if c == 27:                   # ESC 退出
        break
```

#### 逐帧解析

```python
ret, frame = cap.read()
```

|返回值|类型|含义|
|---|---|---|
|ret|bool|True = 成功读取一帧，False = 视频结束/读取失败|
|frame|NumPy array|当前帧的图像数据（和 imread 读到的格式完全一样）|

#### 循环退出的两个条件

|条件|含义|
|---|---|
|ret is not True|视频播放完毕，无帧可读|
|c == 27|用户按 ESC 主动退出|

### 5\. waitKey\(\) 参数与播放速度

```python
c = cv.waitKey(10)    # 等待 10ms
```

这个 10 决定了什么？

```Plain Text
每帧间隔 = waitKey 值（ms）
waitKey(10)：每帧显示 10ms → 理论最大 100 帧/秒
waitKey(33)：每帧显示 33ms → 约 30 帧/秒（模拟 30fps 视频播放）
waitKey(1)： 每帧 1ms → 尽快播放（受限于解码速度）
```

|waitKey 值|效果|
|---|---|
|1|最快速度播放|
|33|模拟 30fps 播放|
|0|每帧都暂停，按任意键切下一帧（逐帧模式）|

如果不加 waitKey\(\)，imshow\(\) 没机会刷新，窗口会卡死。

### 6\. 摄像头 vs 视频文件的差异

```python
# 视频文件
cap = cv.VideoCapture("D:/TV sucai/01.mp4")
Frame = cv.flip(frame, 1)    # ❌ 不需要镜像
# 摄像头
cap = cv.VideoCapture(0)
Frame = cv.flip(frame, 1)    # ✅ 通常需要左右镜像（像照镜子一样自然）
```

|差异|视频文件|摄像头|
|---|---|---|
|需要镜像|❌ 不需要|✅ 常见（自拍体验更自然）|
|帧率|固定（取决于文件）|取决于摄像头硬件|
|结束条件|读完所有帧 → ret=False|永不结束（实时流）|
|可回放|✅ 可以 seek 到任意帧|❌ 只能实时|

### 7\. 完整模板：视频/摄像头处理框架

```python
# 打开视频源
cap = cv.VideoCapture("video.mp4")    # 或 0（摄像头）
while True:
    ret, frame = cap.read()           # 读取一帧
    if not ret:
        break                         # 视频结束
    # ====== 在这里处理 frame ======
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = cv.flip(frame, 1)       # 摄像头时启用
    # ==============================
    cv.imshow("frame", frame)         # 显示
    if cv.waitKey(10) == 27:          # ESC 退出
        break
cap.release()                          # 释放资源
cv.destroyAllWindows()
```

⚠️ 代码中没有显式调用 cap\.release\(\)，但良好的习惯加上它——释放视频资源，避免文件被占用。

## 三、两个函数对比

|维度|rotate\_demo\(\)|video\_demo\(\)|
|---|---|---|
|核心功能|任意角度旋转图像|读取并显示视频/摄像头|
|核心 API|getRotationMatrix2D \+ warpAffine|VideoCapture \+ cap\.read\(\)|
|数学基础|三角函数、仿射变换矩阵|无|
|循环方式|一次性处理|while True 逐帧循环|
|退出方式|waitKey(0\) 按任意键|waitKey(10\) \+ ESC|
|输入|单张图片|视频文件（或摄像头流）|
|输出|单张旋转后的图片|连续的视频帧显示|

## 四、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|np\.radians\(\)|角度 → 弧度|Day 11 ✅|
|np\.cos\(\) / np\.sin\(\)|三角函数|Day 11 ✅|
|abs\(\)|取绝对值|Day 11 ✅|
|cv\.getRotationMatrix2D\(\)|获取 2D 旋转矩阵|Day 11 ✅|
|cv\.warpAffine\(\)|执行仿射变换|Day 11 ✅|
|cv\.VideoCapture\(\)|打开视频文件或摄像头|Day 11 ✅|
|cap\.read\(\)|读取一帧|Day 11 ✅|
|cap\.get\(\)|获取视频属性（宽/高/帧率）|Day 11 ✅|
|cv\.CAP\_PROP\_FRAME\_WIDTH|视频宽度属性|Day 11 ✅|
|cv\.CAP\_PROP\_FRAME\_HEIGHT|视频高度属性|Day 11 ✅|
|cv\.CAP\_PROP\_FPS|视频帧率属性|Day 11 ✅|

## 五、明天计划

视频处理与保存

图像直方图