import numpy as np

#决策树
class Node:
    """决策树节点类"""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 分裂特征索引
        self.threshold = threshold          # 分裂阈值
        self.left = left                    # 左子树
        self.right = right                  # 右子树
        self.value = value                  # 叶节点类别（若为叶节点）

class DecisionTree:
    """分类决策树类"""
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth              # 最大树深
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.root = None                        # 树根节点

    def fit(self, X, y):
        """训练决策树"""
        self.n_features_ = X.shape[1]  # 特征数量
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        """递归构建树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 停止条件：达到最大深度、最小样本数或纯节点
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            return Node(value=self._most_common_label(y))

        # 寻找最佳分裂
        best_feature, best_threshold = self._best_split(X, y, np.arange(n_features))
        if best_feature is None:  # 无有效分裂
            return Node(value=self._most_common_label(y))

        # 分裂数据
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        left_X, left_y = X[left_idxs], y[left_idxs]
        right_X, right_y = X[right_idxs], y[right_idxs]

        # 递归构建子树
        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feature_indices):
        """寻找最佳特征和阈值"""
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])  # 候选阈值，每一个阈值进行的分裂点都要计算，找出最优的阈值，进行分裂
            for threshold in thresholds:
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs
                if sum(left_idxs) == 0 or sum(right_idxs) == 0:
                    continue
                gain = self._gini_gain(y, left_idxs, right_idxs)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini_gain(self, y, left_idxs, right_idxs):
        """计算基尼增益"""
        n = len(y)
        n_left, n_right = sum(left_idxs), sum(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0
        parent_gini = self._gini(y)
        left_gini = self._gini(y[left_idxs])
        right_gini = self._gini(y[right_idxs])
        weighted_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        return parent_gini - weighted_gini

    def _gini(self, y):
        """计算基尼指数"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        """返回多数类别"""
        return np.bincount(y).argmax()

    def predict(self, X):
        """预测样本类别"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """遍历树预测单样本"""
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, depth=0, feature_names=None):
        """打印树结构"""
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}Predict: Class {node.value}")
        else:
            feature = feature_names[node.feature_index] if feature_names else f"Feature {node.feature_index}"
            print(f"{indent}{feature} <= {node.threshold:.2f}")
            print(f"{indent}Left:")
            self.print_tree(node.left, depth + 1, feature_names)
            print(f"{indent}Right:")
            self.print_tree(node.right, depth + 1, feature_names)