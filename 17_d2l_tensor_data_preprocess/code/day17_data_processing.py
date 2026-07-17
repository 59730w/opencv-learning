#数据预处理
import os
import pandas as pd
import torch

# 手动创建数据集文件夹与csv文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取csv数据
data = pd.read_csv(data_file)
print("原始数据：")
print(data)

# 分离输入特征与输出标签
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 数值列缺失值填充均值
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print("\n均值填充缺失值后：")
print(inputs)

# 类别特征独热编码
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print("\n独热编码后：")
print(inputs)

# 转换为torch张量
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print("\n特征张量X：")
print(X)
print("\n标签张量y：")
print(y)