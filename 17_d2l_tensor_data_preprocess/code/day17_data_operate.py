# 数据操作基础
import torch

x = torch.arange(12)
print("x:\n", x)

print("\nx.shape:", x.shape)

print("\nx.numel():", x.numel())

X = x.reshape(3, 4)
print("\nX.reshape(3,4):\n", X)

zero_tensor = torch.zeros((2, 3, 4))
print("\ntorch.zeros((2,3,4)):\n", zero_tensor)

one_tensor = torch.ones((2, 3, 4))
print("\ntorch.ones((2,3,4)):\n", one_tensor)

rand_tensor = torch.randn(3, 4)
print("\ntorch.randn(3,4):\n", rand_tensor)

manual_tensor = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("\n手动创建张量:\n", manual_tensor)

# 张量四则运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print("\nx+y, x-y, x*y, x/y, x**y:")
print(x + y, x - y, x * y, x / y, x ** y)

# 指数运算
print("\ntorch.exp(x):\n", torch.exp(x))

# 张量拼接
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
cat_dim0 = torch.cat((X, Y), dim=0)
cat_dim1 = torch.cat((X, Y), dim=1)
print("\n按行拼接dim=0:\n", cat_dim0)
print("\n按列拼接dim=1:\n", cat_dim1)

# 条件比较
print("\nX == Y:\n", X == Y)

# 求和
print("\nX.sum():", X.sum())

# 广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print("\na:\n", a)
print("\nb:\n", b)
print("\na + b:\n", a + b)

# 索引与切片
print("\nX[-1], X[1:3]:")
print(X[-1], "\n", X[1:3])

# 单元素赋值
X[1, 2] = 9
print("\n修改X[1,2]后:\n", X)

# 批量赋值
X[0:2, :] = 12
print("\n批量赋值后X:\n", X)

# 原地操作验证
before = id(Y)
Y = Y + X
print("\nY重新赋值后地址是否变化:", id(Y) == before)

# 原地赋值 preserve id
Z = torch.zeros_like(Y)
print('\n修改前id(Z):', id(Z))
Z[:] = X + Y
print('修改后id(Z):', id(Z))

# 自增原地操作
before = id(X)
X += Y
print("\nX += Y 后地址是否不变:", id(X) == before)

# tensor 与 numpy 转换
A = X.numpy()
B = torch.tensor(A)
print("\ntype(A), type(B):", type(A), type(B))

# 标量提取
a = torch.tensor([3.5])
print("\na, a.item(), float(a), int(a):")
print(a, a.item(), float(a), int(a))

# ========== 课后练习 ==========
print("\n========== 课后练习1：条件比较 X > Y / X < Y ==========")
# 重置原始张量
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("X > Y:\n", X > Y)
print("\nX < Y:\n", X < Y)

print("\n========== 课后练习2：三维张量广播测试 ==========")
# 三维广播测试
m = torch.randn(2, 1, 4)
n = torch.randn(1, 3, 4)
print("三维张量 m(2,1,4) + n(1,3,4) 结果形状:", (m + n).shape)
print("三维广播结果:\n", m + n)