#线性代数
import torch

# 标量运算
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print("x + y, x * y, x / y, x**y")
print(x + y, x * y, x / y, x**y)

# 向量基础
x = torch.arange(4)
print("\nx =", x)
print("x[3] =", x[3])
print("len(x) =", len(x))
print("x.shape =", x.shape)

# 矩阵与转置
A = torch.arange(20).reshape(5, 4)
print("\nA:\n", A)
print("A.T:\n", A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("\nB:\n", B)
print("B == B.T:\n", B == B.T)

# 三维张量
X = torch.arange(24).reshape(2, 3, 4)
print("\n三维张量 X:\n", X)

# 矩阵按元素运算
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print("\nA:\n", A)
print("A + B:\n", A + B)
print("A * B:\n", A * B)

# 标量与张量广播
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print("\na + X:\n", a + X)
print("(a * X).shape =", (a * X).shape)

# 求和操作
x = torch.arange(4, dtype=torch.float32)
print("\nx =", x)
print("x.sum() =", x.sum())

print("\nA.shape =", A.shape)
print("A.sum() =", A.sum())

A_sum_axis0 = A.sum(axis=0)
print("\nA.sum(axis=0):", A_sum_axis0, "shape:", A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print("A.sum(axis=1):", A_sum_axis1, "shape:", A_sum_axis1.shape)

print("A.sum(axis=[0,1]) =", A.sum(axis=[0, 1]))

# 均值
print("\nA.mean() =", A.mean(), "A.sum()/A.numel() =", A.sum() / A.numel())
print("A.mean(axis=0):", A.mean(axis=0), "A.sum(axis=0)/A.shape[0]:", A.sum(axis=0) / A.shape[0])

# 保持维度求和
sum_A = A.sum(axis=1, keepdims=True)
print("\nsum_A:\n", sum_A)
print("A / sum_A:\n", A / sum_A)

# 累积求和
print("\nA.cumsum(axis=0):\n", A.cumsum(axis=0))

# 点积
y = torch.ones(4, dtype=torch.float32)
print("\nx =", x, "y =", y)
print("torch.dot(x,y) =", torch.dot(x, y))
print("torch.sum(x * y) =", torch.sum(x * y))

# 矩阵向量乘法
print("\nA.shape =", A.shape, "x.shape =", x.shape)
print("torch.mv(A, x):\n", torch.mv(A, x))

# 矩阵乘法
B = torch.ones(4, 3)
print("\ntorch.mm(A, B):\n", torch.mm(A, B))

# 范数
u = torch.tensor([3.0, -4.0])
print("\ntorch.norm(u) =", torch.norm(u))
print("torch.abs(u).sum() =", torch.abs(u).sum())
print("torch.norm(torch.ones((4,9))) =", torch.norm(torch.ones((4, 9))))