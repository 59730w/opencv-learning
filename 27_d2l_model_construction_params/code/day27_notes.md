# Day27：模型构造 \+ 参数管理 \+ 自定义层 \+ 读写文件

**核心主题**：`nn.Module` 内部机制 → 参数访问/管理/初始化/共享 → 自定义无参数层和带参数层 → 模型保存与加载

---

# 第一部分：模型构造

## 1\.1 Sequential：快速搭建

Sequential 是最常用的「积木式」搭建方式，网络层按顺序串行执行：

```python
net = nn.Sequential(
    nn.Linear(20, 256),   # 全连接：20维输入 → 256维输出
    nn.ReLU(),            # 激活函数
    nn.Linear(256, 10)    # 全连接：256维输入 → 10维输出
)
X = torch.rand(2, 20)     # 2个样本，每个样本20维特征
net(X)                    # 输出shape: (2, 10)
```

## 1\.2 继承 nn\.Module 自定义网络

Sequential 仅支持顺序执行的网络结构。若需要**跳跃连接、多分支、条件控制流**等灵活结构，必须手动继承 `nn.Module` 自定义网络。

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()                     # ① 必须调用父类初始化方法
        self.hidden = nn.Linear(20, 256)       # ② 在初始化中定义所有子网络层
        self.out = nn.Linear(256, 10)
    def forward(self, X):
        # ③ 在forward中定义前向传播计算逻辑
        return self.out(F.relu(self.hidden(X)))
```

|方法|职责|必备操作|
|---|---|---|
|`__init__()`|声明网络子模块/层结构|必须调用 `super().__init__()`|
|`forward(X)`|定义前向传播计算流程|返回最终输出张量|

🔑**核心原则**：`__init__` 定义网络结构，`forward` 定义计算流程。PyTorch 会自动追踪`__init__` 中注册的 `nn.Module` 子类，统一管理参数、梯度和设备。

## 1\.3 实现 MySequential——底层机制理解

手动实现 Sequential，理解 `_modules` 核心机制：

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 遍历所有传入的网络层，注册到内部有序字典
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module   
    def forward(self, X):
        # 按注册顺序依次执行前向传播
        for block in self._modules.values():
            X = block(X)
        return X
```

|组件|含义|
|---|---|
|`*args`|接收任意数量的网络子层|
|`self._modules`|nn\.Module 内置**有序字典**，专门存储所有子模块|
|`str(idx)`|以字符串序号作为字典key，保证执行顺序|

🧠 **关键区别**：直接 `self.xxx = layer` 也可赋值，但只有注册到 `_modules` 的层，才能被 `net.parameters()`、`net.to(device)` 等方法正常管理。nn\.Module 内置 `__setattr__` 魔法方法，赋值 Module 实例时会自动注册到 `_modules`。

测试代码：

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X).shape)  # torch.Size([2, 10])
```

功能与官方 `nn.Sequential` 完全一致，这就是 Sequential 的底层实现原理。

## 1\.4 含固定权重和控制流的网络

nn\.Module 极具灵活性，`forward` 中可编写任意 Python 逻辑（固定权重、层复用、循环控制流等）：

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 固定权重，不参与训练
        self.linear = nn.Linear(20, 20)  # 可训练权重层
    def forward(self, X):
        X = self.linear(X)
        # 固定权重矩阵运算，无梯度更新
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)               # 同一网络层复用两次
        # 自定义循环控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()                   # 返回标量结果
```

|技巧|核心代码|应用场景|
|---|---|---|
|固定权重|`requires_grad=False`|预训练嵌入、位置编码、固定变换|
|层权重复用|同一层实例多次调用|孪生网络、权重共享场景|
|自定义控制流|while/if 动态逻辑|自适应迭代、动态计算步数|

## 1\.5 嵌套组合网络

nn\.Module 支持**层层嵌套**，可组合自定义模块与官方标准模块，搭建复杂层次化网络：

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.linear = nn.Linear(32, 16)
    def forward(self, X):
        return self.linear(self.net(X))

# 混合嵌套组装复杂网络
chimera = nn.Sequential(
    NestMLP(),            # 自定义子网络
    nn.Linear(16, 20),    # 官方标准层
    FixedHiddenMLP()      # 含控制流的自定义网络
)
```

🎯 **模块化设计哲学**：每个 nn\.Module 子类都是独立黑盒，外部仅需关注输入输出维度，内部逻辑可任意复杂，即 PyTorch「乐高式编程」。

---

# 第二部分：参数管理

## 2\.1 参数访问基础

```python
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net[2].state_dict()   # 返回有序字典：{参数名: 参数张量}
net[2].bias           # 直接访问偏置属性，返回 nn.Parameter 对象
net[2].bias.data      # 取出底层纯张量数据
```

|访问方式|返回类型|用途|
|---|---|---|
|`.state_dict()`|OrderedDict|模型保存、加载、参数迁移|
|`.weight/.bias`|nn\.Parameter|查看参数形状、梯度信息|
|`.weight.data`|Tensor|手动读写参数数值|

## 2\.2 Parameter vs Tensor vs data

```python
type(net[2].bias)        # <class 'torch.nn.parameter.Parameter'>
type(net[2].bias.data)   # <class 'torch.Tensor'>
net[2].weight.grad is None  # True（未执行反向传播，无梯度）
```

|类型/属性|说明|
|---|---|
|`nn.Parameter`|Tensor子类，自动被 `net.parameters()` 收集，参与梯度更新|
|`.data`|返回底层原始张量，脱离 autograd 梯度追踪|
|`.grad`|梯度存储位置，未执行 `backward()` 时恒为 None|

⚠️ 训练前梯度一定为 None，若不为空说明梯度未清零，会造成参数更新错误。

## 2\.3 参数遍历：named\_parameters\(\) vs parameters\(\)

```python
# 遍历单层参数
[(name, param.shape) for name, param in net[0].named_parameters()]
# [('weight', torch.Size([8, 4])), ('bias', torch.Size([8]))]

# 递归遍历全网所有参数
[(name, param.shape) for name, param in net.named_parameters()]
# [('0.weight', torch.Size([8, 4])),
#  ('0.bias',   torch.Size([8])),
#  ('2.weight', torch.Size([1, 8])),
#  ('2.bias',   torch.Size([1]))]
```

|函数|返回值|使用场景|
|---|---|---|
|`.parameters()`|纯参数张量列表（无名称）|传入优化器更新参数|
|`.named_parameters()`|\(参数名, 参数张量\) 元组列表|调试、分层初始化、参数筛选|
|`.state_dict()`|有序参数字典|模型保存与加载|

命名规则：Sequential 网络中，第 N 个子模块参数前缀为 `N.`，权重固定命名 `weight`，偏置固定命名 `bias`。

## 2\.4 嵌套网络的参数访问

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))

# 嵌套参数层级访问
rgnet[0][1][0].bias.data
#  ↑    ↑  ↑  ↑
#  |    |  |  └── block1 的第 0 层 Linear(4,8)
#  |    |  └──── block2 的第 1 个子模块
#  |    └─────── rgnet 的第 0 个子模块（block2）
#  └──────────── 最外层网络
```

`add_module(name, module)`：动态命名添加子模块，等价于 `self._modules[name] = module`。

## 2\.5 参数初始化

### 2\.5\.1 全局统一初始化 net\.apply\(fn\)

`net.apply()` 会深度优先遍历所有子模块，批量执行自定义初始化函数。

```python
# 正态分布初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 权重 N(0, 0.01²)
        nn.init.zeros_(m.bias)                        # 偏置置0
net.apply(init_normal)
```

### 2\.5\.2 常数初始化

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)  # 所有权重固定为1
        nn.init.zeros_(m.bias)          # 偏置固定为0
```

### 2\.5\.3 分层差异化初始化

```python
net[0].apply(init_xavier)    # 浅层使用Xavier初始化
net[2].apply(init_42)        # 输出层自定义常数初始化
```

🎯 实战常用策略：浅层用 Xavier/Kaiming 初始化保证梯度稳定，分类头用小方差初始化避免输出震荡。

### 2\.5\.4 自定义复杂初始化逻辑

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)          # 均匀分布初始化
        m.weight.data *= m.weight.data.abs() >= 5    # 仅保留绝对值≥5的权重，生成稀疏权重
net.apply(my_init)
```

📌 支持数学运算自定义初始化，可生成稀疏权重、限定权重范围等特殊效果。

## 2\.6 直接手动修改参数

```python
net[0].weight.data[:] += 1         # 所有权重整体+1
net[0].weight.data[0, 0] = 42      # 修改单个参数数值
```

⚠️ 必须通过 `.data` 修改参数，直接赋值 `net[0].weight = xxx` 会破坏 nn\.Parameter 注册机制，导致参数无法被管理。

## 2\.7 参数共享

同一网络层实例多次复用，实现参数共享：

```python
shared = nn.Linear(8, 8)   # 定义单个可复用层
net = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),
    shared, nn.ReLU(),     # 第一次复用
    shared, nn.ReLU(),     # 第二次复用（与上一处完全共享参数）
    nn.Linear(8, 1)
)

# 验证参数共享
print(torch.all(net[2].weight.data == net[4].weight.data))
net[2].weight.data[0, 0] = 100
print(torch.all(net[2].weight.data == net[4].weight.data)) # 同步变化
```

|特性|说明|
|---|---|
|内存共享|多处复用指向同一块参数内存|
|梯度累加|多次前向传播的梯度会自动累加至同一参数|
|应用场景|孪生网络、Transformer嵌入层、权重共享模型|

---

# 第三部分：自定义层

## 3\.1 无参数自定义层

无需可训练参数，仅做固定张量变换：

```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()   # 张量去中心化，均值归零

# 测试效果
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
# tensor([-2., -1.,  0.,  1.,  2.])

# 嵌入网络使用
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())  # 趋近于0（浮点精度误差 1e-8）
```

🎯 常见无参数层：标准化、维度变换、归一化、固定数学运算层。

## 3\.2 带参数自定义全连接层

手动注册 `nn.Parameter`，实现可训练自定义网络层：

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # 手动注册可训练参数
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias   = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        # 前向传播计算
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

|步骤|核心要点|
|---|---|
|参数注册|必须用 `nn.Parameter()` 包装，才能被优化器收集更新|
|\.data 取值|教学演示用法，跳过梯度追踪；实战可直接使用参数本身|

## 3\.3 堆叠自定义层

自定义层与官方标准层接口完全统一，可自由嵌套堆叠：

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)).shape)  # torch.Size([2, 1])
```

---

# 第四部分：读写文件

## 4\.1 保存/加载单个张量

```python
x = torch.arange(4)
torch.save(x, 'x-file')       # 保存张量到文件
x2 = torch.load('x-file')     # 加载张量
print(x2)  # tensor([0, 1, 2, 3])
```

## 4\.2 保存/加载张量列表

```python
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')  # 直接解包加载
```

## 4\.3 保存/加载字典

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
```

`torch.save` 基于 pickle 序列化，支持所有Python可序列化对象：张量、列表、字典、参数模型等。

## 4\.4 模型参数保存与加载（核心重点）

工业级标准用法：仅保存参数，不保存网络结构，轻量化、兼容性强。

```python
# 定义网络结构
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

# 保存模型参数
net = MLP()
X = torch.randn(2, 20)
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')

# 加载模型参数
clone = MLP()                # 1. 先重建相同结构网络
clone.load_state_dict(torch.load('mlp.params'))  # 2. 载入参数
clone.eval()                 # 3. 切换推理模式

# 验证参数一致性
Y_clone = clone(X)
print(torch.all(Y_clone == Y))  # True
```

|保存方式|文件内容|推荐度|
|---|---|---|
|`torch.save(net.state_dict())`|纯参数字典|✅ 推荐，跨设备、跨版本兼容|
|`torch.save(net)`|完整模型对象|❌ 不推荐，依赖原代码结构，迁移性差|

## 4\.5 state\_dict\(\) 本质

`state_dict()` 仅存储参数名与参数张量的映射，**不存储网络结构**，因此加载前必须先创建同结构的网络实例。

```python
net.state_dict()
# OrderedDict([
#     ('hidden.weight', tensor([...])),
#     ('hidden.bias',   tensor([...])),
#     ('output.weight', tensor([...])),
#     ('output.bias',   tensor([...]))
# ])
```

---

# 第五部分：四大模块逻辑总览

模型构造 → 定义网络结构、生成可训练参数

    │

    ▼

    参数管理 → 参数访问、初始化、共享、梯度管理

    │

    ▼

    自定义层 → 扩展官方层能力，自定义无/带参数网络模块

    │

    ▼
读写文件 → 保存训练参数、断点续训、模型迁移部署

---

# 第六部分：核心代码模版速查

## 6\.1 自定义网络通用模版

```python
class MyNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.layer1 = nn.Linear(...)
        self.layer2 = nn.Linear(...)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
```

## 6\.2 参数初始化通用模版

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
net.apply(init_weights)
```

## 6\.3 自定义层通用模版

```python
class MyLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias   = nn.Parameter(torch.zeros(out_dim))
    def forward(self, X):
        return X @ self.weight + self.bias
```

## 6\.4 模型保存/加载通用模版

```python
# 保存模型参数
torch.save(model.state_dict(), 'checkpoint.pth')

# 加载模型参数
model = MyNet()                      # 重建网络结构
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()                         # 切换推理模式
```

---

# 第七部分：本日关键记忆点

|编号|知识点|核心一句话总结|
|---|---|---|
|①|`super().__init__()`|继承nn\.Module必须首行调用，否则子模块无法被追踪管理|
|②|`_modules` 有序字典|PyTorch自动拦截子模块赋值，统一注册管理所有网络层|
|③|`net.apply(fn)`|深度优先遍历全网，批量执行初始化、自定义操作|
|④|参数共享|同一层实例多次复用，参数、梯度全局同步更新|
|⑤|nn\.Parameter|可训练参数专用类型，自动被优化器收集更新|
|⑥|\.data 与 Parameter|\.data是纯数据无梯度追踪，训练优先直接使用Parameter对象|
|⑦|state\_dict 保存机制|仅保存参数不存结构，加载必须先重建相同网络|
|⑧|weight 与 weight\.data|前者是带梯度的参数对象，后者是仅用于手动修改的纯张量|
