# Day4：TrackBar 亮度对比度调节 \+ 键盘响应操作

## 一、TrackBar 进阶 — 同时调节亮度与对比度

### 回顾 Day 3

Day 3 只用了一个 TrackBar 调节亮度：

```python
pos = cv.getTrackbarPos("lightness", "input")
blank[:, :] = (pos, pos, pos)
result = cv.add(image, blank)              # 只改了亮度
```

Day 4 的核心升级：两个 TrackBar 同时控制亮度和对比度，用 **cv\.addWeighted\(\)** 一步完成。

### 1\. cv\.addWeighted\(\) — 图像加权混合

```python
result = cv.addWeighted(image, contrast, blank, 0.5, lightness)
```

#### 函数签名

```python
cv.addWeighted(src1, alpha, src2, beta, gamma)
```

#### 计算公式

```Plain Text
dst = src1 × alpha + src2 × beta + gamma
```

|参数|含义|本例取值|
|---|---|---|
|src1|第一张图像|image（原图）|
|alpha|src1 的权重|contrast（TrackBar/100，范围 1\.0\~2\.0）|
|src2|第二张图像|blank（全零图像）|
|beta|src2 的权重|0\.5（固定值）|
|gamma|亮度偏移量|lightness（TrackBar值，范围 0\~100）|

#### 本代码中的实际效果

因为 blank 是全零图像（所有像素 = 0），所以：

```Plain Text
result = image × contrast + 0 + lightness
       = image × contrast + lightness
```

|参数|TrackBar 范围|实际值|效果|
|---|---|---|---|
|contrast|100\~200|1\.0\~2\.0|拖大 → 对比度增强，色彩更鲜明|
|lightness|0\~100|0\~100|拖大 → 整体变亮|

#### 为什么 contrast 初始值设为 100？

```python
cv.createTrackbar("contrast", "input", 100, 200, nothing)
```

初始值 100，除以 100 = 1\.0，即原始对比度不变。拖到 200 时 = 2\.0，对比度翻倍。

### 2\. TrackBar 参数传递的完整流程

```Plain Text
① 创建窗口    → cv.namedWindow()
② 创建滑动条  → cv.createTrackbar() × 2
③ 显示原图    → cv.imshow("input", image)
④ 进入循环
   ├── 读取两个滑块值    → cv.getTrackbarPos() × 2
   ├── contrast 除以 100  → 1.0~2.0
   ├── 加权混合          → cv.addWeighted()
   ├── 显示结果          → cv.imshow()
   └── 检测 ESC          → break
⑤ 销毁窗口    → cv.destroyAllWindows()
```

## 二、键盘响应操作

### 核心理念

cv\.waitKey\(\) 不只是暂停——它返回按键的 ASCII 码。利用这个返回值，我们可以让不同的按键触发不同的图像处理。

### 1\. cv\.waitKey\(\) 返回值

```python
c = cv.waitKey(1)    # 等待 1ms，返回按键的 ASCII 码
```

- 没有按键时返回 \-1

- 有按键时返回 该键的 ASCII 码（整数）

### 2\. ASCII 码速查表（常用）

|按键|ASCII 码|说明|
|---|---|---|
|0 \~ 9|48 \~ 57|数字键|
|A \~ Z|65 \~ 90|大写字母|
|a \~ z|97 \~ 122|小写字母|
|ESC|27|退出键|
|Space|32|空格键|
|Enter|13|回车键|

⚠️ 代码中不要写 c == '1'（那是字符比较），要写 c == 49（ASCII 码整数比较）。

### 3\. 代码中的按键映射

```python
while True:
    c = cv.waitKey(1)
    if c == 49:            # 按键 1 → 灰度图
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imshow("result", gray)
    if c == 50:            # 按键 2 → HSV 图
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        cv.imshow("result", hsv)
    if c == 51:            # 按键 3 → 取反
        opposite = cv.bitwise_not(image)
        cv.imshow("result", opposite)
    if c == 27:            # 按键 ESC → 退出
        break
```

|按键|触发操作|API|
|---|---|---|
|1|显示灰度图|cv\.cvtColor(\.\.\., COLOR\_BGR2GRAY\)|
|2|显示HSV图|cv\.cvtColor(\.\.\., COLOR\_BGR2HSV\)|
|3|显示取反图|cv\.bitwise\_not\(\)|
|ESC|退出|break|

#### 设计模式

```Plain Text
while True:
    c = waitKey(1)           ← 轻量级等待，持续循环
    if c == 某键: 做某操作   ← 按键检测
    if c == 27: break        ← 退出条件
```

这种模式是 OpenCV 交互程序的标准写法，后续项目中会反复用到。

### 4\. cv\.bitwise\_not\(\) — 位运算取反

```python
opposite = cv.bitwise_not(image)
```

这是 Day 4 第一次接触位运算

效果等同于 Day 2 的逐像素取反（255 \- 每个通道值）

但 bitwise\_not 是 C\+\+ 底层实现的向量化运算，速度远超 Python 双重循环

|方式|速度|代码量|
|---|---|---|
|Day 2：双重循环逐像素取反|极慢|多|
|Day 4：cv\.bitwise\_not\(\)|极快|一行|

这就是 OpenCV 的核心价值——用 C\+\+ 底层替你做了高效实现，Python 接口一行搞定。

## 三、两个函数对比总结

|维度|adjust\_contrast\_demo\(\)|keys\_demo\(\)|
|---|---|---|
|交互方式|TrackBar 滑块|键盘按键|
|调整内容|亮度 \+ 对比度（连续）|图像类型切换（离散）|
|核心 API|cv\.addWeighted\(\) \+ cv\.getTrackbarPos\(\)|cv\.waitKey\(\) 返回值 \+ ASCII 码判断|
|退出方式|按 ESC|按 ESC|
|适用场景|参数调优（实时预览）|功能切换（快捷操作）|

## 四、今日学到的 API

|API|作用|首次出现|
|---|---|---|
|cv\.addWeighted\(\)|图像加权混合，同时控制亮度与对比度|Day 4 ✅|
|cv\.getTrackbarPos\(\)|获取 TrackBar 当前值|Day 3|
|cv\.createTrackbar\(\)|创建 TrackBar|Day 3|
|cv\.bitwise\_not\(\)|按位取反（像素取反）|Day 4 ✅|
|cv\.waitKey\(\) 返回值|获取按键 ASCII 码，实现键盘交互|Day 4 ✅|

## 五、明天计划

图像平滑滤波：均值滤波、高斯滤波、中值滤波

去噪 \+ 边缘保留
