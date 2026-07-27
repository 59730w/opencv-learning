# 读写文件
import torch
from torch import nn
from torch.nn import functional as F

# 保存与加载张量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print("x2 =", x2)

# 保存张量列表
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print("\nx2, y2 =", (x2, y2))

# 保存字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print("\nmydict2 =", mydict2)

# 模型参数保存与加载
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 保存参数
torch.save(net.state_dict(), 'mlp.params')

# 加载参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

Y_clone = clone(X)
print("\nY_clone == Y:\n", Y_clone == Y)