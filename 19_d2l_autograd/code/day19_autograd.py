#自动求导
import torch

# 1. 创建向量
x = torch.arange(4.0)
print("x =", x)

# 开启梯度跟踪，查看初始梯度
x.requires_grad_(True)
print("初始x.grad:", x.grad)

# 构建目标函数 y=2x·x
y = 2 * torch.dot(x, x)
print("y =", y)

# 反向传播求梯度
y.backward()
print("x.grad =", x.grad)
print("x.grad == 4 * x:", x.grad == 4 * x)

# 梯度清零，测试sum求导
x.grad.zero_()
y = x.sum()
y.backward()
print("\n梯度清零后，y=x.sum()求导 x.grad =", x.grad)

# 非标量输出反向传播
x.grad.zero_()
y = x * x
y.sum().backward()
print("y=x*x sum反向传播 x.grad =", x.grad)

# detach分离计算图
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print("\nx.grad == u:", x.grad == u)

# 不使用detach重新求导
x.grad.zero_()
y.sum().backward()
print("x.grad == 2 * x:", x.grad == 2 * x)

# 含控制流的求导函数
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print("\na.grad == d / a:", a.grad == d / a)

# ====================== 课后练习 ======================
print("\n========== 课后练习演示 ==========")
# 练习2：反向传播后再次执行backward
print("\n练习2：重复执行backward测试")
x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print("第一次反向传播 x.grad:", x.grad)
try:
    y.backward()
except RuntimeError as e:
    print("再次backward报错：", e)

# 练习3：a改为随机向量测试控制流梯度
print("\n练习3：a改为向量测试控制流梯度")
a_vec = torch.randn(3, requires_grad=True)
d_vec = f(a_vec)
d_vec.backward()
print("向量a梯度：", a_vec.grad)
print("d/a 逐元素：", d_vec / a_vec)

# 练习4：自定义控制流梯度示例
print("\n练习4：自定义控制流求导案例")
def custom_func(x):
    out = x
    for _ in range(3):
        if out.sum() < 5:
            out = out + x
        else:
            out = out * x
    return out
t = torch.tensor([1.0, 2.0], requires_grad=True)
res = custom_func(t)
res.sum().backward()
print("自定义函数梯度 t.grad =", t.grad)

# 练习5：绘制sin(x)及其导数图像
import matplotlib.pyplot as plt
print("\n练习5：绘制sin(x)与自动求导导函数")
x_plot = torch.linspace(-torch.pi, torch.pi, 100, requires_grad=True)
f = torch.sin(x_plot)
# 反向传播求导
f.sum().backward()
df = x_plot.grad

plt.figure(figsize=(8, 4))
plt.plot(x_plot.detach(), f.detach(), label="f(x)=sin(x)")
plt.plot(x_plot.detach(), df.detach(), label="df/dx (autograd)")
plt.legend()
plt.grid(True)
plt.show()