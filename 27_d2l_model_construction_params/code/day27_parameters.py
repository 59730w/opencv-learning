# 参数管理
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print("net[2].state_dict():")
print(net[2].state_dict())

print("\ntype(net[2].bias):", type(net[2].bias))
print("net[2].bias:", net[2].bias)
print("net[2].bias.data:", net[2].bias.data)

print("\nnet[2].weight.grad == None:", net[2].weight.grad is None)

print("\n第一层参数名称与形状:")
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print("\n全部参数名称与形状:")
print(*[(name, param.shape) for name, param in net.named_parameters()])

print("\nnet.state_dict()['2.bias'].data:")
print(net.state_dict()['2.bias'].data)

# 嵌套网络
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print("\n嵌套网络结构：")
print(rgnet)

print("\nrgnet[0][1][0].bias.data:")
print(rgnet[0][1][0].bias.data)

# 正态初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print("\n正态初始化结果 net[0].weight.data[0], net[0].bias.data[0]:")
print(net[0].weight.data[0], net[0].bias.data[0])

# 常数初始化
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print("\n常数初始化结果 net[0].weight.data[0], net[0].bias.data[0]:")
print(net[0].weight.data[0], net[0].bias.data[0])

# 分层不同初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print("\nxavier初始化第一层权重：")
print(net[0].weight.data[0])
print("\n常数42初始化最后一层权重：")
print(net[2].weight.data)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print("\n自定义初始化 net[0].weight[:2]:")
print(net[0].weight[:2])

# 直接修改参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print("\n手动修改后 net[0].weight.data[0]:")
print(net[0].weight.data[0])

# 共享参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
print("\n共享层权重初始是否相等：")
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print("修改一处后两处是否同步变化：")
print(net[2].weight.data[0] == net[4].weight.data[0])