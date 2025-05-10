import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# 模拟混合数据
X = np.array([[1.0, 'red'], [2.0, 'blue'], [3.0, 'red'], [4.0, 'green']])
y = np.array([0, 1, 0, 1])

# 创建 ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), [0]),        # 数值列标准化
    ('cat', OneHotEncoder(), [1])          # 分类列独热编码
])

# 创建 Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),        # 预处理
    ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))  # 分类器
])

# 训练 Pipeline
pipeline.fit(X, y)

# 评估
accuracy = pipeline.score(X, y)
print(f"Pipeline accuracy: {accuracy:.4f}")