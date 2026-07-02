# Day2：图像对象创建与赋值 \+ 像素读写操作

## 一、图像的本质：NumPy 数组

OpenCV 中图像就是一个 NumPy 三维数组：

```python
image = cv.imread("D:/images/test.jpg")
print(image.shape)   # (h, w, c)  →  高度、宽度、通道数
print(image)         # 打印所有像素值
```

|维度|含义|BGR 彩色图|灰度图|
|---|---|---|---|
|shape\[0\]|高度 h（行数）|如 800|如 800|
|shape\[1\]|宽度 w（列数）|如 600|如 600|
|shape\[2\]|通道数 c|3（BGR）|无（不显示）|

BGR 每个通道取值范围：0\~255

读取灰度图：cv\.imread("路径", cv\.IMREAD\_GRAYSCALE\) → shape 只有 \(h, w\)

## 二、图像对象的创建

图像本质上就是数组，所以创建图像 = 创建 NumPy 数组。

### 方式一：np\.zeros\_like\(\) — 创建全黑图像

```python
blank = np.zeros_like(image)
```

创建一个和 image 尺寸相同、全为 0 的数组

全 0 = 纯黑色图像

### 方式二：np\.copy\(\) — 完整拷贝

```python
blank = np.copy(image)
```

把 image 的全部数据复制一份给 blank

修改 blank 不影响原图

### 方式三：np\.zeros\(\) — 手动指定尺寸

```python
blank = np.zeros((480, 640, 3), dtype=np.uint8)
```

创建一个 480×640 的 3 通道黑色图像

## 三、图像对象的赋值

赋值语法（和 NumPy 切片一样）

```python
blank[60:500, 60:500, :] = image[60:500, 60:500, :]
```

|部分|含义|
|---|---|
|60:500|第 60 行到第 499 行（高度方向）|
|60:500|第 60 列到第 499 列（宽度方向）|
|:|所有通道（B、G、R 全复制）|

⚠️ 重要：如果是灰度图

灰度图没有通道维度，所以切片不要写最后的 :：

```python
gray = cv.imread("D:/images/test.jpg", cv.IMREAD_GRAYSCALE)
roi = gray[60:500, 60:500]          # ✅ 正确，两个维度
roi = gray[60:500, 60:500, :]       # ❌ 错误，灰度图没有第三维
```

## 四、ROI（Region of Interest）感兴趣区域

```python
roi = image[60:500, 60:500, :]
```

ROI = 从原图中截取一块矩形区域

语法：image[起始行:结束行, 起始列:结束列, 通道\]

ROI 和原图共享数据（NumPy 视图机制），修改 ROI 会影响原图

**ROI 的应用场景（与森林监测相关）**

从无人机航拍图中截取感兴趣区域

提取单株树冠区域做后续分析

缩小处理范围，减少计算量

## 五、图像像素的读写操作

### 读像素

```python
b, g, r = image[row, col]
```

image[row, col\] 返回一个包含 3 个值 的数组：[B, G, R\]

注意顺序是 B、G、R，不是 R、G、B

灰度图只返回一个值：gray\_val = gray[row, col\]

### 写像素

```python
image[row, col] = (b_val, g_val, r_val)
```

赋一个元组，按 (B, G, R\) 顺序

例如设为红色：image[row, col\] = (0, 0, 255\)

### 遍历所有像素

```python
h, w, c = image.shape
for row in range(h):        # 遍历每一行
    for col in range(w):    # 遍历每一列
        b, g, r = image[row, col]           # 读
        image[row, col] = (255-b, 255-g, 255-r)  # 写（取反）
```

h：图像高度 = 多少行像素

w：图像宽度 = 多少列像素

注意：双重循环很慢，大图尽量不要用。能用 NumPy 向量化操作就不要用循环。

## 六、像素取反的实现

```python
b, g, r = image[row, col]
image[row, col] = (255-b, 255-g, 255-r)
```

原理：每个通道用 255 减去原始值

效果：亮的变暗，暗的变亮 → 得到负片效果

例如白色 \(255,255,255\) → 变成黑色 \(0,0,0\)

例如红色 \(0,0,255\) → 变成青色 \(255,255,0\)

## 七、代码中的两个函数总结

|函数|功能|关键知识点|
|---|---|---|
|mat\_demo\(\)|创建空白图像 \+ 截取 ROI \+ 赋值|np\.zeros\_like\(\)、切片赋值、ROI|
|pixel\_demo\(\)|逐像素遍历 \+ 取反|image\.shape、像素读写、双重循环|

## 八、今日学到的 API / 概念

|API / 概念|作用|
|---|---|
|image\.shape|获取图像尺寸 \(h, w, c\)|
|np\.zeros\_like\(image\)|创建同尺寸的全零数组|
|np\.copy\(image\)|完整拷贝图像|
|image[y1:y2, x1:x2, :\]|ROI 区域截取与赋值|
|image[row, col\]|读取单个像素|
|image[row, col\] = (b,g,r\)|写入单个像素|
|cv\.IMREAD\_GRAYSCALE|以灰度模式读取图像|

## 九、明天计划

图像算术运算：加法、加权混合

图像逻辑运算：与、或、非、异或

亮度与对比度调整

