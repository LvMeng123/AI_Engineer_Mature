#MLP模型
import torch
import torch.nn as nn               #神经网络模块
import torch.nn.functional as F     #包含激活函数的函数式接口
import numpy as np
import sklearn.preprocessing as pre

class MLP_Mnist(nn.Module):
    def __init__(self, input_size=784, hidden_size1 = 128, hidden_size2 = 64, num_classes=10):
        super(MLP_Mnist, self).__init__()
        
        #定义层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x

data = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]])

#标准化数据
standardModel = pre.StandardScaler()
data_scaler = standardModel.fit_transform(data)
print(data_scaler)

