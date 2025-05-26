import os
import torch 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class SEHE_Kaggle:
    def __init__(self):
        self.path = 'E:\机器学习\Kaggle竞赛\sehe-4678-ai-202'
        self.filename = os.listdir(self.path)
        print(self.filename)
        self.train_data = pd.read_csv(os.path.join(self.path, self.filename[2]))
        self.test_data = pd.read_csv(os.path.join(self.path, self.filename[1]))
    
    def pictureRelationship(self):
        print('训练数据信息', self.train_data.info())
        print('训练数据前五行', self.train_data[:5])
        print(self.train_data.describe())
        
        #绘制数据分析图
        #目标数据分析
        plt.figure(figsize=(10,6)) #设置图像大小
        plt.hist(self.train_data['MEDV'], bins='auto',color='black')
        plt.title('MEDV MIddle Price', fontsize=15)
        plt.xlabel('MEDV', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', alpha=0.35)  #添加水平网格，alpha是透明度
        plt.show()
        
        #特征数据分析
        features_to_plt = ['RM', 'LSTAT', 'CRIM']
        plt.figure(figsize=(15,5))
        
        for i, feature in enumerate(features_to_plt):
            plt.subplot(1, len(features_to_plt), i+1)
            plt.hist(self.train_data[feature], bins='auto', color='black')
            plt.title(f'{feature}', fontsize=13)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('frequency', fontsize=12)
            plt.grid(axis='y', alpha=0.75)
            
        plt.tight_layout() #自读调整子图参数，使其填充整个图像区域
        plt.show()
        
        #特征与目标值之间的关系图
        plt.figure(figsize=(18,5))
        for i, feature in enumerate(features_to_plt):
            plt.subplot(1, len(features_to_plt), i+1)
            plt.scatter(self.train_data[feature], self.train_data['MEDV'], alpha=0.6, edgecolors='k', s = 20)
            plt.title('Relationship of feature and MEDV', fontsize=13)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('MEDV', fontsize=12)
            plt.grid(True)        
        
        plt.tight_layout() #自读调整子图参数，使其填充整个图像区域
        plt.show()
        
        #热力图，特征之间的相关性
        # 计算特征之间的相关性矩阵
        # 我们这里包含了目标变量 MEDV，这样也可以看到各个特征与 MEDV 的相关系数
        correlation_matrix = self.train_data.corr()

        plt.figure(figsize=(12, 10)) # 设置图像大小

        # 使用 seaborn 的 heatmap 功能绘制热力图
        # annot=True 会在每个格子上显示数值
        # cmap='coolwarm' 是一种常用的颜色映射方案，红色表示正相关，蓝色表示负相关
        # fmt='.2f' 表示数值保留两位小数
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('特征相关性热力图 (Correlation Matrix)', fontsize=15)
        plt.show()
    
    def process_Data(self):
        target_coloum = "MEDV"
        X = self.train_data.drop(target_coloum, axis=1)
        y = self.train_data[target_coloum].values.reshape(-1, 1)  # 将目标变量转换为二维数组
        print('X.head', X.head())
        # print('y.head', y.head())
        
        #划分数据集,数据还没有转换，仍然是np，pd形式
        random_seed = np.random.seed(42)
        test_set_ratio = 0.2
        X_train, X_val, y_train, y_val = train_test_split(X.values, y, test_size=test_set_ratio, random_state=random_seed)
        
        #特征缩放
        scale_X = StandardScaler()
        scale_y = StandardScaler()
        X_train_scale_np = scale_X.fit_transform(X_train)
        X_val_scaled = scale_X.transform(X_val)
        y_train_scale_np = scale_y.fit_transform(y_train)
        y_val_scaled = scale_y.transform(y_val)
        
        #转换成FloatTensor
        X_train_tensor= torch.tensor(X_train_scale_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scale_np, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
        
        #构造dataset
        X_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        X_val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        #构造dataloader
        batch_size = 64
        train_loader = DataLoader(X_train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(X_val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, scale_X, scale_y # 返回 scaler_y 以便后续反向转换
