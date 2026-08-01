# VGG
import torch
import matplotlib
matplotlib.use('TkAgg')
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
device = d2l.try_gpu()

train_loss_list = []
train_acc_list = []
test_acc_list = []

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    metric = d2l.Accumulator(3)
    net.train()
    for X, y in train_iter:
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat,y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat,y), X.shape[0])
    train_l = metric[0]/metric[2]
    train_acc = metric[1]/metric[2]
    test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
    train_loss_list.append(train_l)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f'epoch {epoch+1}, loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

epochs = list(range(1, num_epochs+1))
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(epochs, train_loss_list, label="train loss")
ax.plot(epochs, train_acc_list, label="train acc")
ax.plot(epochs, test_acc_list, label="test acc")
ax.set_xlabel("epoch")
ax.legend()
plt.savefig("vgg_manual_curve.png", dpi=300, bbox_inches="tight")
print("✅图片已保存 vgg_manual_curve.png")