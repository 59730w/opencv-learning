# Windows完整版PyTorch（GPU）\+d2l深度学习环境安装教程

## 一、环境整体配置信息

- 适配系统：Windows 10 / Windows 11

- Conda安装路径：**D:\\conda**

- 虚拟环境名称：**d2l\-zh**

- Python版本：3\.9

- PyTorch版本：GPU版（CUDA 12\.1）

- 项目代码路径：**D:\\DL\_code\\d2l\-zh**

- 适用教程：《动手学深度学习》（d2l\-zh）

## 二、前置准备工作

### 1\. 安装Miniconda

优先选择Miniconda（轻量化、无冗余软件），安装核心要求：

- 自定义安装路径为：**D:\\conda**（禁止安装在C盘）

- 安装界面勾选：**Add Miniconda to my PATH environment variable**（自动配置环境变量）

### 2\. 准备课程代码

下载d2l\-zh官方配套代码包，解压后放入路径：**D:\\DL\_code\\d2l\-zh**，无中文、无空格路径。

## 三、创建Conda虚拟环境

1\. 打开系统终端：Anaconda Prompt (miniconda3\)

2\. 执行命令创建Python3\.9专属虚拟环境

```Plain Text
conda create -n d2l-zh python=3.9
```

3\. 终端提示 `Proceed ([y]/n)?` 输入 `y` 回车，等待环境创建完成

4\. 激活虚拟环境（每次使用环境需先激活）

```Plain Text
conda activate d2l-zh
```

✅ 激活成功标识：终端前缀由 `(base)` 变为 `(d2l-zh)`

## 四、安装GPU版PyTorch（核心）

在 `(d2l-zh)` 环境下，执行官方CUDA12\.1 GPU加速版本安装命令，支持NVIDIA显卡硬件加速：

```Plain Text
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

等待全部依赖下载、安装完成，无报错即为成功。

## 五、安装d2l课程配套库

使用清华镜像源加速安装，适配动手学深度学习全部案例代码：

```Plain Text
pip install d2l -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 六、PyCharm环境配置（最终运行环境）

规避Jupyter SSL系统底层报错，全程使用PyCharm运行课程代码，稳定无bug

### 1\. 导入项目文件夹

PyCharm → 文件 → 打开 → 选择路径 **D:\\DL\_code\\d2l\-zh** → 信任项目

### 2\. 配置虚拟环境解释器（关键步骤）

1\. 文件 → 设置 → 项目 → Python解释器 → 右上角【添加】

2\. 左侧选择：**系统解释器**（避开conda识别bug）

3\. 浏览选择解释器绝对路径：**D:\\conda\\envs\\d2l\-zh\\python\.exe**

4\. 依次点击：确定 → 应用 → 关闭，环境配置完成

## 七、环境完整性验证

### 方式1：PyCharm代码验证

新建/打开任意py文件，粘贴测试代码并运行：

```Plain Text
import torch
import d2l
# 打印PyTorch版本
print("PyTorch版本：", torch.__version__)
# 验证GPU是否成功启用
print("GPU加速可用：", torch.cuda.is_available())
```

✅ 正常结果：输出版本号 \+ **GPU加速可用：True**

### 方式2：PyCharm终端验证

1\. 打开PyCharm底部Terminal终端

2\. 激活环境：`conda activate d2l-zh`

3\. 输入 `python` 进入Python交互界面，执行上方测试代码即可

## 八、关键问题说明

- Jupyter报错说明：本机存在**Windows系统证书库损坏**底层bug，无法修复，无需折腾Jupyter

- 替代方案：PyCharm完全替代Jupyter，支持代码分段运行、绘图、调试、GPU加速，不影响全部课程学习

- 环境持久性：所有环境、依赖均安装在D盘，关闭终端、重启电脑不会丢失配置

- 禁止操作：不要修改中文路径、不要切换base环境、不要重复安装依赖

## 九、环境最终状态总结

✅ Conda虚拟环境正常创建 \| ✅ Python3\.9环境适配

✅ GPU版PyTorch安装成功、硬件加速生效

✅ d2l课程库完整安装 \| ✅ PyCharm环境完美适配

✅ 可直接运行所有动手学深度学习案例代码