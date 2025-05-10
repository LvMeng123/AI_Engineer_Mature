from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 假设 dt_classifier (Wine 分类树) 已经训练好了
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train_w, y_train_w) # 确保用浅层树

print("\n--- 可视化决策树 (Wine 分类树, max_depth=3) ---")
plt.figure(figsize=(20, 12)) # 设置图像大小，否则可能太小看不清
plot_tree(dt_classifier,
          feature_names=wine.feature_names, # 特征名称
          class_names=wine.target_names,    # 类别名称 (分类树用)
          filled=True,        # 用颜色填充节点以表示纯度/类别
          rounded=True,       # 节点边框用圆角
          fontsize=10)        # 字体大小
plt.title("Decision Tree for Wine Classification (max_depth=3)", fontsize=16)
plt.show()

# 对于回归树 dt_regressor
# dt_regressor = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train_d, y_train_d)
# plt.figure(figsize=(18,10))
# plot_tree(dt_regressor,
#           feature_names=diabetes.feature_names,
#           filled=True,
#           rounded=True,
#           fontsize=9)
# plt.title("Decision Tree for Diabetes Regression (max_depth=3)", fontsize=16)
# plt.show()