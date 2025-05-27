import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import zscore # 用于Z-score异常值移除
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split # KFold 和 train_test_split
from sklearn.metrics import mean_squared_error

# --- （A）数据处理类 ---
# 你可以将这个逻辑整合到你现有的 SEHE_Kaggle 类中，或者使用一个新的类
class DataProcessor:
    def __init__(self, data_directory_path, train_filename='train.csv', test_filename='test_without_labels.csv'):
        self.path = data_directory_path
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.scaler_X = StandardScaler() # X的缩放器，将在训练数据上拟合
        self._load_data()

    def _load_data(self):
        try:
            self.train_df_raw = pd.read_csv(os.path.join(self.path, self.train_filename))
            self.test_df_raw = pd.read_csv(os.path.join(self.path, self.test_filename))
            print("原始训练数据和测试数据加载成功。")
        except FileNotFoundError as e:
            print(f"错误: 数据文件未找到 - {e}")
            self.train_df_raw = None
            self.test_df_raw = None
            raise # 重新抛出异常，因为后续无法进行

    def get_processed_data_for_kfold_and_submission(self, target_column="MEDV"):
        if self.train_df_raw is None or self.test_df_raw is None:
            print("原始数据未加载，无法处理。")
            return None, None, None

        # 复制数据以避免修改原始DataFrame
        train_df = self.train_df_raw.copy()
        test_df = self.test_df_raw.copy()

        print("\n--- 开始数据预处理 ---")
        # 1. 特征剔除
        cols_to_drop = ['NOX', 'B'] # 和"更好代码"一致
        train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
        test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
        print(f"已剔除特征: {cols_to_drop}")

        # 2. 分离特征和目标 (仅对训练集)
        y_train_series = train_df[target_column]
        X_train_df = train_df.drop(columns=[target_column], errors='ignore')

        # 确保测试集和训练集有相同的特征列 (在剔除NOX, B和target之后)
        # 这是为了处理Kaggle测试集可能多出或缺少ID列等情况
        train_features = X_train_df.columns.tolist()
        # 保留测试集中与训练特征一致的列，并按训练特征的顺序排列
        X_test_df = test_df[train_features].copy()


        # 3. Z-score 异常值移除 (仅对训练集X)
        # 计算Z-scores时只使用数值型特征
        numeric_cols_train = X_train_df.select_dtypes(include=np.number).columns
        z_scores = np.abs(zscore(X_train_df[numeric_cols_train]))
        # 保留所有特征的Z-score都小于3的行
        keep_indices = (z_scores < 3).all(axis=1)
        X_train_df_no_outliers = X_train_df[keep_indices]
        y_train_series_no_outliers = y_train_series[keep_indices] # 对应过滤y
        print(f"通过Z-score移除了 {len(X_train_df) - len(X_train_df_no_outliers)} 个训练集中的异常值行。")


        # 4. 目标变量 log1p 转换 (仅对训练集y)
        y_train_log_transformed = np.log1p(y_train_series_no_outliers.values)
        print("训练集目标变量已进行 log1p 转换。")

        # 5. 特征标准化 (StandardScaler)
        # 在处理完异常值的训练特征上拟合scaler_X
        X_train_scaled = self.scaler_X.fit_transform(X_train_df_no_outliers.values)
        # 用同一个scaler_X转换测试集特征
        X_test_scaled = self.scaler_X.transform(X_test_df.values) # 注意这里用的是 X_test_df
        print("训练集和测试集特征已进行标准化。")
        
        print("--- 数据预处理完毕 ---")
        # y_train_log_transformed 已经是 NumPy 数组
        return X_train_scaled, y_train_log_transformed, X_test_scaled


# --- （B）MLP 模型定义 (更灵活的版本) ---
class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size=1, dropout_rate=0.2, activation_name='relu'):
        super(FlexibleMLP, self).__init__()
        
        layers = nn.ModuleList()
        current_dim = input_size

        # 选择激活函数
        if activation_name.lower() == 'relu':
            activation_fn = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation_fn = nn.LeakyReLU()
        elif activation_name.lower() == 'elu':
            activation_fn = nn.ELU()
        elif activation_name.lower() == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation_name}")

        # 构建隐藏层
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size)) # 加入BatchNorm
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_size
        
        self.hidden_network = nn.Sequential(*layers) # 将所有隐藏层组件放入Sequential
        self.output_layer = nn.Linear(current_dim, output_size) # 输出层

        print(f"FlexibleMLP 初始化: input={input_size}, hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}, activation={activation_name}, dropout={dropout_rate}")

    def forward(self, x):
        x = self.hidden_network(x)
        x = self.output_layer(x)
        return x

# --- （C）主训练和评估逻辑 ---
def run_training_pipeline():
    # --- 0. 初始化和全局设置 ---
    DATA_DIR = r'E:\机器学习\Kaggle竞赛\sehe-4678-ai-202' # **请修改为你的实际路径**
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test_without_labels.csv' # 确保文件名正确
    SAMPLE_SUBMISSION_FILE = 'sample submission.csv' # 确保文件名正确
    TARGET_COLUMN = 'MEDV'
    
    # 设置随机种子以保证可复现性
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 下面两行可以增强确定性，但可能略微影响性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 1. 数据预处理 ---
    data_processor = DataProcessor(data_directory_path=DATA_DIR, train_filename=TRAIN_FILE, test_filename=TEST_FILE)
    # X_scaled 是整个训练集的缩放后特征, y_log是整个训练集的log1p转换后目标, X_submission_scaled 是测试集的缩放后特征
    X_scaled, y_log, X_submission_scaled = data_processor.get_processed_data_for_kfold_and_submission(target_column=TARGET_COLUMN)

    if X_scaled is None:
        return

    # --- 2. K-Fold 交叉验证设置 ---
    n_splits = 8 # 和"更好代码"一致
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_val_losses_mse = [] # 存储每一折在K-Fold验证集上的MSE损失 (log尺度)
    oof_val_losses_rmse_orig = [] # 存储每一折在K-Fold验证集上的RMSE (原始尺度)
    submission_predictions_folds = [] # 存储每一折对测试集的预测 (log尺度)
    
    fold_histories = [] # 存储每一折的训练历史，用于后续画图

    print(f"\n--- 开始 {n_splits}-折交叉验证 ---")
    total_training_start_time = time.time()

    for fold, (train_fold_idx, val_fold_idx) in enumerate(kfold.split(X_scaled, y_log)):
        fold_start_time = time.time()
        print(f"\n--- 第 {fold+1}/{n_splits} 折 ---")

        # 当前折的训练数据和K-Fold验证数据 (已缩放/转换)
        X_train_fold_data = X_scaled[train_fold_idx]
        y_train_fold_data = y_log[train_fold_idx].reshape(-1, 1) # 确保是 (n, 1)
        X_val_fold_data = X_scaled[val_fold_idx]
        y_val_fold_data = y_log[val_fold_idx].reshape(-1, 1)

        # **在当前折的训练数据内部再划分出一小部分作为“内部验证集”用于早停和模型选择**
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
            X_train_fold_data, y_train_fold_data, test_size=0.2, random_state=seed # 例如20%作为内部验证
        )

        # 转换为 PyTorch 张量
        X_train_inner_tensor = torch.tensor(X_train_inner, dtype=torch.float32)
        y_train_inner_tensor = torch.tensor(y_train_inner, dtype=torch.float32)
        X_val_inner_tensor = torch.tensor(X_val_inner, dtype=torch.float32)
        y_val_inner_tensor = torch.tensor(y_val_inner, dtype=torch.float32)
        
        X_val_fold_tensor = torch.tensor(X_val_fold_data, dtype=torch.float32).to(device) # K-Fold验证集特征
        y_val_fold_tensor = torch.tensor(y_val_fold_data, dtype=torch.float32).to(device) # K-Fold验证集目标 (log尺度)


        train_inner_dataset = TensorDataset(X_train_inner_tensor, y_train_inner_tensor)
        val_inner_dataset = TensorDataset(X_val_inner_tensor, y_val_inner_tensor)

        train_inner_loader = DataLoader(train_inner_dataset, batch_size=16, shuffle=True) # 可调参数
        val_inner_loader = DataLoader(val_inner_dataset, batch_size=16, shuffle=False)

        # 初始化模型
        input_size = X_train_inner.shape[1]
        model = FlexibleMLP(input_size=input_size, 
                            hidden_size=128,       # 可调参数
                            num_hidden_layers=3,   # 可调参数
                            dropout_rate=0.2,      # 可调参数
                            activation_name='elu'  # 可调参数
                           ).to(device)

        criterion = nn.MSELoss() # 损失在 log(1+y) 尺度上计算
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01) # 可调参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

        # 训练当前折的模型
        num_epochs_fold = 100 # 可调参数
        best_inner_val_loss = float('inf')
        best_model_fold_path = f"best_model_fold_{fold+1}.pth"
        
        current_fold_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs_fold):
            model.train()
            running_train_loss = 0.0
            for inputs, labels in train_inner_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
            
            epoch_train_loss = running_train_loss / len(train_inner_loader.dataset)
            current_fold_history['train_loss'].append(epoch_train_loss)

            # 内部验证
            model.eval()
            running_inner_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_inner_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_inner_val_loss += loss.item() * inputs.size(0)
            
            epoch_inner_val_loss = running_inner_val_loss / len(val_inner_loader.dataset)
            current_fold_history['val_loss'].append(epoch_inner_val_loss)
            scheduler.step(epoch_inner_val_loss)

            if (epoch + 1) % 20 == 0: # 每20轮打印一次进度
                print(f"  Fold {fold+1}, Epoch [{epoch+1}/{num_epochs_fold}], Train Loss: {epoch_train_loss:.4f}, Inner Val Loss: {epoch_inner_val_loss:.4f}")

            if epoch_inner_val_loss < best_inner_val_loss:
                best_inner_val_loss = epoch_inner_val_loss
                torch.save(model.state_dict(), best_model_fold_path)
        
        fold_histories.append(current_fold_history) # 保存当前 fold 的训练历史

        # 加载当前折的最佳模型，在K-Fold的验证集上评估
        model.load_state_dict(torch.load(best_model_fold_path))
        model.eval()
        with torch.no_grad():
            # 预测是在 log(1+y) 尺度
            preds_val_fold_log = model(X_val_fold_tensor)
            # 计算损失也是在 log(1+y) 尺度
            loss_val_fold_mse_log = criterion(preds_val_fold_log, y_val_fold_tensor).item()
            oof_val_losses_mse.append(loss_val_fold_mse_log)

            # 为了计算原始尺度的RMSE，需要反向转换
            preds_val_fold_original = np.expm1(preds_val_fold_log.cpu().numpy())
            y_val_fold_original = np.expm1(y_val_fold_tensor.cpu().numpy())
            rmse_val_fold_original = np.sqrt(mean_squared_error(y_val_fold_original, preds_val_fold_original))
            oof_val_losses_rmse_orig.append(rmse_val_fold_original)
        
        print(f"  Fold {fold+1} 结束。K-Fold Val MSE(log): {loss_val_fold_mse_log:.4f}, K-Fold Val RMSE(orig): {rmse_val_fold_original:.4f}")
        print(f"  Fold {fold+1} 耗时: {time.time() - fold_start_time:.2f}s")

        # 使用当前折的最佳模型对测试集进行预测
        with torch.no_grad():
            X_submission_tensor = torch.tensor(X_submission_scaled, dtype=torch.float32).to(device)
            fold_submission_preds_log = model(X_submission_tensor).cpu().numpy()
            submission_predictions_folds.append(fold_submission_preds_log)
        
    total_training_time = time.time() - total_training_start_time
    print(f"\n--- {n_splits}-折交叉验证结束 ---")
    print(f"平均 K-Fold 验证 MSE (log尺度): {np.mean(oof_val_losses_mse):.4f} (+/- {np.std(oof_val_losses_mse):.4f})")
    print(f"平均 K-Fold 验证 RMSE (原始尺度): {np.mean(oof_val_losses_rmse_orig):.4f} (+/- {np.std(oof_val_losses_rmse_orig):.4f})")
    print(f"总训练耗时: {total_training_time:.2f} 秒")

    # --- 3. 生成提交文件 ---
    # 对测试集的预测取所有 fold 的平均值 (在 log 尺度上平均)
    avg_submission_preds_log = np.mean(submission_predictions_folds, axis=0)
    # 反向转换为原始尺度
    final_submission_preds = np.expm1(avg_submission_preds_log)

    # 读取 sample submission 文件以获取格式
    try:
        submission_df = pd.read_csv(os.path.join(DATA_DIR, SAMPLE_SUBMISSION_FILE))
        # 假设提交文件中的预测列名与你的目标列名或 'y_pred' 等一致
        # 如果Kaggle提供的测试集有ID列，而你的test_df_raw第一列就是ID，可以用它
        # submission_df[TARGET_COLUMN] = final_submission_preds.flatten() # flatten() 以防是 (N,1)
        # 检查 sample submission 的列名，通常是 'id' 和一个预测列，如 'MEDV' 或 'y_pred'
        # 假设 sample submission 的预测列名是 'MEDV'
        if TARGET_COLUMN in submission_df.columns:
            submission_df[TARGET_COLUMN] = final_submission_preds.flatten()
        elif 'y_pred' in submission_df.columns: # 有些竞赛用 y_pred
             submission_df['y_pred'] = final_submission_preds.flatten()
        else:
            # 如果不确定，可以创建一个新的DataFrame
            print("警告: sample_submission.csv 中未找到明确的目标列名。将使用 'y_pred'。")
            # 假设 test_df_raw 的第一列是ID，如果你的测试集ID是独立的，则需要另外加载
            ids_test = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE)).iloc[:, 0] # 假设ID是第一列
            submission_df = pd.DataFrame({'id': ids_test, 'MEDV': final_submission_preds.flatten()})


        submission_df.to_csv("submission.csv", index=False)
        print("\n提交文件 'submission.csv' 已生成。")
        print(submission_df.head())
    except FileNotFoundError:
        print(f"错误: sample submission 文件 '{SAMPLE_SUBMISSION_FILE}' 未在 '{DATA_DIR}' 中找到。无法生成提交文件。")


    # --- 4. 可视化 (例如，第一个fold的训练/内部验证损失) ---
    if fold_histories:
        best_fold_idx = np.argmin(oof_val_losses_rmse_orig) # 找到RMSE最低的fold
        print(f"\n为表现最佳的 Fold {best_fold_idx+1} 绘制训练/内部验证损失曲线 (其 K-Fold Val RMSE(orig): {oof_val_losses_rmse_orig[best_fold_idx]:.4f})")
        
        best_fold_history = fold_histories[best_fold_idx]
        epochs_fold_range = range(1, len(best_fold_history['train_loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_fold_range, best_fold_history['train_loss'], label='该 Fold 的训练损失', marker='.')
        plt.plot(epochs_fold_range, best_fold_history['val_loss'], label='该 Fold 的内部验证损失', marker='.')
        plt.title(f'Fold {best_fold_idx+1} 的训练/内部验证损失 (log尺度)', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('MSE Loss (log尺度)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"fold_{best_fold_idx+1}_train_val_loss_curves.png")
        plt.show()

if __name__ == '__main__':
    run_training_pipeline()