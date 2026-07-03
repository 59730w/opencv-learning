# Day3：图像算术运算 \+ TrackBar 动态调整亮度

## 一、图像算术运算

图像算术运算就是对**每个像素的每个通道**做数学运算。

### 核心原理

- 图像 = NumPy 数组，算术运算 = 对数组中每个元素做运算

- 所有运算都是**逐像素、逐通道**进行的

- 结果值超出 0\~255 范围时会被**饱和截断**（OpenCV 函数）或**取模**（NumPy 运算符）

## 二、OpenCV 算术运算函数

### 1\. 创建常量图像

```python
blank = np.zeros_like(image)          # 创建和 image 同尺寸的全 0 数组
blank[:, :] = (2, 2, 2)               # 所有像素设为 (B=2, G=2, R=2)
```

|语法|含义|
|---|---|
|blank[:, :\]|选取所有行、所有列|
|blank[:, :\] = (2, 2, 2\)|整个图像的 BGR 三通道都赋值为 2|

### 2\. 加法：cv\.add\(\)

```python
result = cv.add(image, blank)
```

每个像素的每个通道 = image值 \+ blank值

饱和运算：结果 \> 255 → 截断为 255；结果 \< 0 → 截断为 0

效果：图像整体变亮（加了 2，不太明显）

#### ⚠️ cv\.add\(\) vs NumPy \+ 运算符

|方式|超过 255 时|小于 0 时|
|---|---|---|
|cv\.add(a, b\)|截断为 255（饱和）|截断为 0|
|a \+ b（NumPy）|取模（260 → 4）|取模（\-1 → 255）|

这就是为什么 OpenCV 推荐用 cv\.add\(\) 而不是 \+——饱和运算更符合图像处理的直觉。取模会让亮的变暗、暗的变亮，效果奇怪。

### 3\. 减法：cv\.subtract\(\)

```python
result = cv.subtract(image, blank)
```

每个像素值减去常量

饱和运算：结果 \< 0 → 截断为 0

效果：图像整体变暗

### 4\. 乘法：cv\.multiply\(\)

```python
result = cv.multiply(image, blank)
```

每个像素值乘以常量

饱和运算：结果 \> 255 → 截断为 255

效果：对比度大幅提升（亮处更亮、暗处相对不变）

⚠️ 乘数稍微大一点就全白了，一般用小数（如 1\.5）

### 5\. 除法：cv\.divide\(\)

```python
result = cv.divide(image, blank)
```

每个像素值除以常量

效果：图像整体变暗，对比度降低

当 blank 值很小时（如 2），相当于每个通道缩小

### 6\. 四个运算对比总结

|运算|API|效果|饱和截断|
|---|---|---|---|
|加法|cv\.add\(\)|变亮|上限 255|
|减法|cv\.subtract\(\)|变暗|下限 0|
|乘法|cv\.multiply\(\)|对比度 ↑|上限 255|
|除法|cv\.divide\(\)|对比度 ↓|下限 0|

## 三、TrackBar — 动态调整亮度

### 核心理念

TrackBar（滑动条）让你在运行时拖动滑块实时调整参数，不用每次改代码、重新运行。

### 实现步骤

#### 步骤 1：创建窗口 \+ 滑动条

```python
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)                           # 创建窗口
cv.createTrackbar("lightness", "input", 0, 100, nothing)              # 创建滑动条
```

|参数|含义|
|---|---|
|"lightness"|滑动条名称（显示在窗口上）|
|"input"|所属窗口名（必须与 namedWindow 一致）|
|0|初始值|
|100|最大值|
|nothing|回调函数（滑动时触发，这里只打印）|

#### 步骤 2：回调函数

```python
def nothing(x):
    print(x)
```

每当滑动条位置改变，OpenCV 自动调用这个函数

参数 x = 当前滑动条的值

这里只是打印，实际逻辑写在 while 循环里

#### 步骤 3：死循环实时更新

```python
while True:
    pos = cv.getTrackbarPos("lightness", "input")     # 获取当前滑块值
    blank[:, :] = (pos, pos, pos)                      # 用滑块值设置亮度增量
    result = cv.add(image, blank)                      # 加到原图上
    cv.imshow("result", result)                        # 显示结果
    c = cv.waitKey(1)                                  # 等待 1ms
    if c == 27:                                        # ESC 键的 ASCII 码
        break
```

### 关键细节

|要点|说明|
|---|---|
|cv\.getTrackbarPos\(\)|获取当前滑块位置（0\~100）|
|waitKey(1\)|必须设为 1ms，让窗口能刷新，同时能检测按键|
|c == 27|27 = ESC 键的 ASCII 码，按 ESC 退出循环|
|blank[:, :\] = (pos, pos, pos\)|滑块值同时赋给 B、G、R 三通道（灰度增量）|

### 为什么这个设计很巧妙

- 用 blank 存放亮度增量（由 TrackBar 控制）

- 用 cv\.add\(\) 把增量加到原图上

- TrackBar 值从 0 拖到 100 → 每个像素 BGR 同时增加 0\~100

- 拖滑块 = 实时看到图片从暗变亮的效果

## 四、代码结构解析

### math\_demo\(\) 函数

演示四种算术运算（当前执行的是除法）

通过注释切换不同运算，方便对比效果

blank[:, :\] = (2, 2, 2\) 作为运算的另一个操作数

### adjust\_lightness\_demo\(\) 函数

演示 TrackBar 动态交互

namedWindow → createTrackbar → while 循环 → getTrackbarPos → add → imshow

按 ESC 退出循环

## 五、今日学到的 API

|API|作用|
|---|---|
|cv\.add\(\)|图像加法（饱和运算），用于提亮图像，超出255自动截断为255，小于0截断为0|
|cv\.subtract\(\)|图像减法（饱和运算），用于压暗图像，像素值小于0自动截断为0|
|cv\.multiply\(\)|图像乘法（饱和运算），提升图像对比度，亮部更亮，最大值截断255|
|cv\.divide\(\)|图像除法（饱和运算），降低图像对比度，整体压暗画面|
|cv\.namedWindow\(\)|创建自定义命名窗口，用于承载图像、滑动条等交互组件|
|cv\.createTrackbar\(\)|创建可视化滑动条，支持运行时动态调整参数，实现实时交互|
|cv\.getTrackbarPos\(\)|获取滑动条当前实时数值，作为图像处理的动态参数|
|cv\.WINDOW\_AUTOSIZE|窗口自适应图像尺寸，禁止手动缩放，保证画面完整显示|
|waitKey(1\)|窗口实时刷新、监听键盘按键，1ms延迟保证画面流畅更新|
|ESC 键 \(ASCII 27\)|程序退出触发按键，按下即可终止循环、关闭窗口|

## 六、明天计划

- 图像逻辑运算：与、或、非、异或

- 图像位运算的应用（掩码、抠图）
