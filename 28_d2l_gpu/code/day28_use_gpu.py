# GPU
import torch
from torch import nn

# 检测CUDA
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("GPU 数量:", torch.cuda.device_count())
    print("当前设备:", torch.cuda.current_device())
else:
    print("当前Pytorch为CPU版本，无法使用RTX3050显卡")

print("\ntorch.device('cpu'), torch.device('cuda')：")
print(torch.device('cpu'), torch.device('cuda'))

print("\nGPU数量：", torch.cuda.device_count())

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print("\ntry_gpu(), try_gpu(10), try_all_gpus()：")
print(try_gpu(), try_gpu(10), try_all_gpus())

x = torch.tensor([1, 2, 3])
print("\nx.device：", x.device)

X = torch.ones(2, 3, device=try_gpu())
print("\nX：\n", X)

Y = torch.rand(2, 3, device=try_gpu(0))
print("\nY：\n", Y)

Z = X.cuda(0)
print("\nX：\n", X)
print("Z：\n", Z)

print("\nY + Z：\n", Y + Z)

print("\nZ.cuda(0) is Z：", Z.cuda(0) is Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print("\nnet(X)：\n", net(X))

print("\nnet[0].weight.data.device：", net[0].weight.data.device)