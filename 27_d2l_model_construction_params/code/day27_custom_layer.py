# 自定义层
import torch
import torch.nn.functional as F
from torch import nn

# 无参数自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
print("CenteredLayer输出：")
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 嵌入Sequential使用自定义层
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print("\nY均值：", Y.mean())

# 带参数自定义全连接层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print("\n自定义层权重参数：")
print(linear.weight)

print("\n自定义层前向传播输出：")
print(linear(torch.rand(2, 5)))

# 堆叠自定义层构建网络
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print("\n堆叠自定义网络输出：")
print(net(torch.rand(2, 64)))