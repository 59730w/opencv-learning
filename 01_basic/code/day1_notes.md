# Day1：环境搭建 + 图像读取显示 + 色彩空间转换
---
## 一、环境搭建
- Python 环境下安装 OpenCV：
  ```bash
  pip install opencv-python opencv-contrib-python
### 导入方式（两种都可以，cv 更简洁）：

- Python
  ```bash
  import cv2 as cv   # 推荐
  import cv2         # 也可以
测试是否安装成功：在 Python 里输入 print(cv.__version__)，输出 OpenCV 版本号即成功。

## 二、图像读取与显示
1. 读取图像：cv.imread()
- Python
  ```bash
  image = cv.imread("图片路径")
  - 参数：图片路径（绝对路径或相对路径）
  - 返回值：一个 NumPy 数组（三维数组：[高度, 宽度, 通道]）
  - OpenCV 读取时默认是 BGR 色彩顺序（不是 RGB！）
2. 显示图像：cv.imshow()
- Python
  ```bash
  cv.imshow("窗口名称", image)
  cv.waitKey(0)            # 等待按键，0表示无限等待
  cv.destroyAllWindows()   # 关闭所有窗口
  imshow                   # 第一个参数是窗口标题，自己起名
  waitKey()                # 必须写，否则窗口一闪而过，根本看不见
  waitKey(0)：             # 一直等到按任意键
  waitKey(1000)：          # 显示 1 秒后自动关闭
  destroyAllWindows()：    # 养成良好的资源释放习惯
## 三、色彩空间转换：cv.cvtColor()
- Python
  ```bash
  cv.cvtColor(src, 转换类型)
### 常用转换类型

| 转换类型 | 含义 |
| --- | --- |
| `cv.COLOR_BGR2GRAY` | BGR → 灰度图 |
| `cv.COLOR_BGR2HSV` | BGR → HSV |
| `cv.COLOR_BGR2RGB` | BGR → RGB |
| `cv.COLOR_GRAY2BGR` | 灰度 → BGR（三通道） |

#### BGR → 灰度（Gray）
- 灰度图只有一个通道，每个像素值表示亮度（0~255）
- 0 是纯黑，255 是纯白
- **用途**：减少计算量，很多图像处理算法（如边缘检测）只需要灰度图

#### BGR → HSV
- **H（Hue，色调）**：颜色的种类，0~180（OpenCV中）
- **S（Saturation，饱和度）**：颜色的纯度，0~255
- **V（Value，明度）**：亮度，0~255
- **用途**：颜色分割（比如提取绿色植被），用 HSV 比 BGR 直观得多

## 四、代码中的两个关键点
1. if __name__ == "__main__":
    - 这句话的意思是："只有直接运行这个 .py 文件时，才执行下面的代码"
    - 如果这个文件被其他文件 import，if 里面的代码不会执行
    - 这是一个好的编程习惯，写脚本时都加上
2. 为什么 OpenCV 用 BGR 而不是 RGB？
    - 历史原因：早期显卡和摄像头厂商用 BGR 格式，OpenCV 沿用至今
    - 实际影响：用 plt.imshow()（Matplotlib）显示 OpenCV 图片时颜色会反，需要先转成 RGB
- Python
  ```bash
  rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
## 五、今日总结
| 学到的 API | 作用 |
| --- | --- |
| `cv.imread()` | 读取图片，返回 NumPy 数组 |
| `cv.imshow()` | 在窗口中显示图片 |
| `cv.imwrite()` | 保存图片到文件 |
| `cv.waitKey()` | 等待键盘按键 |
| `cv.destroyAllWindows()` | 销毁所有 OpenCV 窗口 |
| `cv.cvtColor()` | 色彩空间转换 |
| `cv.COLOR_BGR2GRAY` | BGR 转灰度 |
| `cv.COLOR_BGR2HSV` | BGR 转 HSV |

## 六、明天计划
- 图像像素的读写与操作
- ROI（感兴趣区域）截取
- 图像算术运算