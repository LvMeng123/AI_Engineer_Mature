import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from SEHE4678_AI_202 import SEHE_Kaggle # 确保你的文件名和类名与此一致
import matplotlib.pyplot as plt # 导入 matplotlib 用于绘图
from sklearn.metrics import mean_squared_error # 用于计算 RMSE

class SEHE4678_Model(nn.Module):
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer=1, dropout_rate=0.2):
        super(SEHE4678_Model, self).__init__()
        self.flattn = nn.Flatten() # 注意：对于已经是 [batch, features] 的输入，这层可能不产生变化
        self.net = nn.Sequential(
            nn.Linear(input_layer, hidden_layer1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer2, output_layer)
        )
        
    def forward(self, x):
        x = self.flattn(x) # 如果输入已经是2D的 [batch_size, num_features]，这步可以省略
        x = self.net(x)
        return x

# --- 主程序开始 ---

# 1. 数据处理
# **修改1：为 SEHE_Kaggle 实例使用不同的变量名，避免与神经网络模型变量名冲突**
data_processor = SEHE_Kaggle()
# 假设 process_Data 正确返回: train_loader, val_loader, 特征缩放器 scale_X, 目标缩放器 scale_y
# 如果你的 process_Data 不缩放y，请确保 scale_y 返回的是 None
try:
    train_loader, val_loader, scale_X, scale_y = data_processor.process_Data()
    # 检查 DataLoader 是否成功创建 (例如，通过检查其长度或尝试获取一个批次)
    if not train_loader or not val_loader:
        raise ValueError("DataLoaders 未成功创建。请检查 process_Data 方法。")
    print(f"数据加载和预处理完成。训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    print(f"输入特征数量 (来自DataLoader): {train_loader.dataset.tensors[0].shape[1]}")
except Exception as e:
    print(f"数据处理过程中发生错误: {e}")
    print("程序将退出。请检查 SEHE_Kaggle 类中的实现和数据文件路径。")
    exit()


# 2. 定义模型超参数和实例化模型
INPUT_LAYER = train_loader.dataset.tensors[0].shape[1]  # 输入层大小
HIDDEN_LAYER1 = 128  # 第一个隐藏层大小 (可以调整)
HIDDEN_LAYER2 = 64   # 第二个隐藏层大小 (可以调整)
DROPOUT_RATE = 0.4   # dropout比率 (可以调整)

model = SEHE4678_Model(input_layer=INPUT_LAYER, 
                       hidden_layer1=HIDDEN_LAYER1, 
                       hidden_layer2=HIDDEN_LAYER2, 
                       dropout_rate=DROPOUT_RATE)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器, lr可以调整

# 4. 训练和验证设置
num_epoch = 100 # 增加训练轮次以观察更完整的学习曲线
best_val_rmse = float('inf')  # 初始化最小验证RMSE为正无穷 (我们将基于RMSE保存模型)
best_model_path = 'best_model_with_validation.pth'  # 最佳模型保存路径

# **修改2：确保设备检查的完整性**
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) # 将模型移动到指定设备
print(f"\n模型已实例化并移动到设备: {device}")
print(model)


history = {'train_loss':[], 'val_loss':[], 'val_rmse_original':[]}
print('\n--- 开始训练 ---')

for epoch in range(num_epoch):
    model.train() # 设置为训练模式
    start_time = time.time()
    running_train_loss = 0.0 # 使用 running_train_loss 累积当前 epoch 的训练损失

    for batch_idx, (features, target) in enumerate(train_loader):
        features, target = features.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, target) # 损失在缩放后的y上计算 (如果y被缩放了)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * features.size(0) 
        
    epoch_train_loss = running_train_loss / len(train_loader.dataset) # 计算当前epoch的平均训练损失
    history['train_loss'].append(epoch_train_loss)
    
    # --- 验证阶段 ---
    model.eval() # 设置为评估模式
    running_val_loss = 0.0
    all_val_preds_scaled_list = []
    all_val_targets_scaled_list = []

    with torch.no_grad(): # 在验证阶段不计算梯度
        for features, target in val_loader:
            features, target = features.to(device), target.to(device)
            outputs = model(features) # 模型预测的是缩放后的y
            loss = criterion(outputs, target) # 损失也在缩放后的y上计算
            running_val_loss += loss.item() * features.size(0)
            
            all_val_preds_scaled_list.append(outputs.cpu())
            all_val_targets_scaled_list.append(target.cpu())
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    history['val_loss'].append(epoch_val_loss)

    # 将收集到的预测和目标连接并转换为 NumPy 数组
    all_val_preds_scaled_np = torch.cat(all_val_preds_scaled_list).numpy()
    all_val_targets_scaled_np = torch.cat(all_val_targets_scaled_list).numpy()
    
    # **重要：如果目标y被缩放了，进行反向转换以计算原始尺度的RMSE**
    if scale_y is not None: # 检查 scale_y 是否是你 process_Data 中返回的有效缩放器
        # 确保形状是 (n_samples, 1) 以便 inverse_transform
        if all_val_preds_scaled_np.ndim == 1:
            all_val_preds_scaled_np = all_val_preds_scaled_np.reshape(-1, 1)
        if all_val_targets_scaled_np.ndim == 1:
            all_val_targets_scaled_np = all_val_targets_scaled_np.reshape(-1, 1)
            
        y_pred_original = scale_y.inverse_transform(all_val_preds_scaled_np)
        y_true_original = scale_y.inverse_transform(all_val_targets_scaled_np)
        # 或者，如果你的 y_val 没有被缩放，可以直接使用 y_val_np (需要从 process_Data 获得)
        # y_true_original = y_val_np # 假设 y_val_np 是原始未缩放的验证集目标
    else: # 如果 y 没有被缩放
        y_pred_original = all_val_preds_scaled_np
        y_true_original = all_val_targets_scaled_np
        # 如果 scale_y 是 None，意味着 y 没有被缩放，所以 "scaled" 值就是原始值

    rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    history['val_rmse_original'].append(rmse_original)
    
    epoch_duration = time.time() - start_time
    print(f'Epoch [{epoch+1}/{num_epoch}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val RMSE: {rmse_original:.4f}, Time: {epoch_duration:.2f}s')
    
    # 保存最佳模型 (基于验证集上的原始RMSE)
    if rmse_original < best_val_rmse:
        best_val_rmse = rmse_original
        torch.save(model.state_dict(), best_model_path)
        print(f"    ** Best model saved to {best_model_path} (Val RMSE: {best_val_rmse:.4f}) **")

print("\n--- 训练结束 ---")
if os.path.exists(best_model_path):
    print(f"最佳模型已保存在: {best_model_path} (对应的最佳验证 RMSE: {best_val_rmse:.4f})")
else:
    print("没有模型被保存（可能训练轮次过少或未达到保存条件）。")


# --- 5. 可视化训练过程 ---
if history['train_loss']: # 确保 history 不为空
    epochs_range = range(1, num_epoch + 1)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1) # 修改为1行2列，因为RMSE通常比损失更重要
    plt.plot(epochs_range, history['train_loss'], label='训练损失 (Train Loss)', marker='.')
    plt.plot(epochs_range, history['val_loss'], label='验证损失 (Validation Loss)', marker='.')
    plt.title('训练和验证损失 (MSE)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_rmse_original'], label='验证集 RMSE (原始尺度)', marker='.', color='green')
    plt.title('验证集 RMSE (原始尺度)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('RMSE ($1000s)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("training_validation_performance_final.png")
    print("\n训练过程曲线图已保存为 'training_validation_performance_final.png'")
    plt.show()
else:
    print("History 为空，无法绘制曲线图。")

# (可选) 加载最佳模型并进行评估
print("\n加载最佳模型进行评估:")
best_model = SEHE4678_Model(input_layer=INPUT_LAYER, hidden_layer1=HIDDEN_LAYER1, hidden_layer2=HIDDEN_LAYER2, dropout_rate=DROPOUT_RATE)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)
best_model.eval()
# # ... 在这里可以用 best_model 在 val_loader 上再次运行评估，或在测试集上预测 ...