import numpy as np
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l

# 生成模拟数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 构造数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print("第一个小批量数据：")
print(next(iter(data_iter)))

# 搭建网络、初始化参数
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 损失函数与优化器
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3

# 训练循环
print("\n训练损失输出：")
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 输出学习到的参数
print("\n学习得到的权重w：", net[0].weight.data)
print("学习得到的偏置b：", net[0].bias.data)