# ---------------------- 1. 基础库导入 ----------------------
from socket import ntohs

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 固定全局随机种子，保证结果可复现
np.random.seed(174)

# ---------------------- 2. 通用可视化函数 ----------------------
def plot_classification_results(model, X, y_true, class_names, title, scaler_mean=None, scaler_std=None):
    """
    绘制分类器的决策边界和预测结果，与真实标签对比
    参数：
        model: 训练好的分类模型
        X: 原始特征数据（4维）
        y_true: 真实标签
        class_names: 类别名称列表
        title: 图表标题
        scaler_mean: 标准化均值（仅SVM需要）
        scaler_std: 标准化标准差（仅SVM需要）
    """
    # 1. 数据预处理（SVM需要标准化）
    if scaler_mean is not None and scaler_std is not None:
        X_scaled = (X - scaler_mean) / scaler_std
    else:
        X_scaled = X

    # 2. PCA降维至2维
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # 3. 生成网格点，用于绘制决策边界
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 4. 将网格点转换回原始特征空间，用于模型预测
    grid_points_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points_pca)

    # 5. 标准化网格点（仅SVM需要）
    if scaler_mean is not None and scaler_std is not None:
        grid_points_scaled = (grid_points_original - scaler_mean) / scaler_std
        Z = model.predict(grid_points_scaled)
    else:
        Z = model.predict(grid_points_original)

    Z = Z.reshape(xx.shape)

    # 6. 获取模型预测结果
    y_pred = model.predict(X_scaled)

    # 7. 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 8. 创建双图对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = 'viridis'  # 统一颜色映射，与K-means一致

    # --- 左图：模型预测结果 + 决策边界 ---
    # 绘制决策边界填充
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    # 绘制样本点
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, s=50, alpha=0.8, edgecolors='k')
    ax1.set_title(f'{title}（预测结果）', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PCA主成分1（方差解释率：{pca.explained_variance_ratio_[0]:.2%}）', fontsize=12)
    ax1.set_ylabel(f'PCA主成分2（方差解释率：{pca.explained_variance_ratio_[1]:.2%}）', fontsize=12)
    # 添加颜色条
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='预测类别', ticks=[0,1,2])
    cbar1.ax.set_yticklabels(class_names)

    # --- 右图：真实标签分布 ---
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap=cmap, s=50, alpha=0.8, edgecolors='k')
    ax2.set_title('鸢尾花真实标签分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PCA主成分1（方差解释率：{pca.explained_variance_ratio_[0]:.2%}）', fontsize=12)
    ax2.set_ylabel(f'PCA主成分2（方差解释率：{pca.explained_variance_ratio_[1]:.2%}）', fontsize=12)
    # 添加颜色条
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='真实类别', ticks=[0,1,2])
    cbar2.ax.set_yticklabels(class_names)

    # 9. 调整布局并显示
    plt.tight_layout()
    plt.show()

# ---------------------- 3. 决策树节点类定义 ----------------------
class Node:
    """树节点类，区分内部决策节点和叶节点"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # 划分特征的索引
        self.threshold = threshold      # 划分阈值
        self.left = left                # 左子树（<=阈值）
        self.right = right              # 右子树（>阈值）
        self.value = value              # 叶节点的分类结果

    def is_leaf_node(self):
        """判断是否为叶节点"""
        return self.value is not None

# ---------------------- 4. 决策树分类器核心类 ----------------------
class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split  # 节点分裂最小样本数
        self.max_depth = max_depth                  # 树的最大深度
        self.n_feats = n_feats                      # 用于分裂的特征数
        self.root = None                             # 树的根节点

    def _gini(self, y):
        """计算基尼系数（文档指定的不纯度指标）"""
        class_count = np.bincount(y)
        class_proba = class_count / len(y)
        return 1 - np.sum([p ** 2 for p in class_proba])

    def _split_dataset(self, X_column, threshold):
        """按特征阈值划分数据集"""
        left_idx = np.argwhere(X_column <= threshold).flatten()
        right_idx = np.argwhere(X_column > threshold).flatten()
        return left_idx, right_idx

    def _find_best_split(self, X, y, feat_idxs):
        """找到基尼系数最小的最优划分"""
        best_gini = float('inf')
        best_feat_idx = None
        best_threshold = None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                left_idx, right_idx = self._split_dataset(X_column, thr)
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                # 计算加权基尼系数
                gini_left = self._gini(y[left_idx])
                gini_right = self._gini(y[right_idx])
                weighted_gini = (len(left_idx)/len(y))*gini_left + (len(right_idx)/len(y))*gini_right
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat_idx = feat_idx
                    best_threshold = thr
        return best_feat_idx, best_threshold

    def _most_common_label(self, y):
        """获取节点中样本数最多的类别"""
        label_count = np.bincount(y)
        return np.argmax(label_count)

    def _grow_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_unique_labels = len(np.unique(y))

        # 递归停止条件
        if (depth >= self.max_depth
            or n_unique_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False) if self.n_feats else np.arange(n_features)
        best_feat, best_thresh = self._find_best_split(X, y, feat_idxs)
        left_idx, right_idx = self._split_dataset(X[:, best_feat], best_thresh)

        left_child = self._grow_tree(X[left_idx, :], y[left_idx], depth+1)
        right_child = self._grow_tree(X[right_idx, :], y[right_idx], depth+1)
        return Node(feature_idx=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        """模型训练"""
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _predict_single_sample(self, x, node):
        """预测单个样本"""
        if node.is_leaf_node():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single_sample(x, node.left)
        else:
            return self._predict_single_sample(x, node.right)

    def predict(self, X):
        """批量预测"""
        return np.array([self._predict_single_sample(x, self.root) for x in X])

# ---------------------- 5. 线性SVM分类器核心类 ----------------------
class LinearSVM:
    def __init__(self, learning_rate=0.001, C=10.0, epochs=10000, verbose=True):
        self.learning_rate = learning_rate  # 优化学习率：步长更小，避免震荡
        self.C = C                          # 优化正则化参数：降低正则化权重，更关注分类错误
        self.epochs = epochs                # 优化迭代次数：保证充分收敛
        self.verbose = verbose              # 打印训练过程
        self.models = {}                    # 存储每个类别的二分类器 {类别: (w, b)}

    def _hinge_loss(self, X, y, w, b):
        """计算合页损失+L2正则化的总损失，用于监控收敛"""
        n_samples = X.shape[0]
        regularization_loss = 0.5 * np.dot(w, w)
        hinge_loss_sum = 0
        for idx in range(n_samples):
            decision_value = np.dot(w, X[idx]) + b
            hinge_loss_sum += max(0, 1 - y[idx] * decision_value)
        total_loss = regularization_loss + self.C * hinge_loss_sum
        return total_loss

    def fit(self, X, y):
        """模型训练：批量梯度下降（BGD），训练更稳定，解决收敛问题"""
        n_samples, n_features = X.shape
        classes = np.unique(y)

        # 为每个类别训练一个独立的二分类器
        for c in classes:
            if self.verbose:
                print(f"\n===== 训练类别 {c} ({['setosa','versicolor','virginica'][c]}) 的二分类SVM =====")
            # 创建二分类标签：当前类别为+1，其他为-1
            y_binary = np.where(y == c, 1, -1)
            # 初始化权重向量w和偏置b
            w = np.zeros(n_features)
            b = 0

            # 批量梯度下降迭代优化
            for epoch in range(self.epochs):
                # 初始化梯度
                dw = np.zeros(n_features)  # 权重的梯度
                db = 0                     # 偏置的梯度

                # 遍历所有样本，累计梯度
                for idx in range(n_samples):
                    decision_value = np.dot(w, X[idx]) + b
                    # 合页损失条件判断
                    if y_binary[idx] * decision_value < 1:
                        # 样本未被正确分类或间隔不足：累计合页损失梯度
                        dw += w - self.C * y_binary[idx] * X[idx]
                        db += -self.C * y_binary[idx]
                    else:
                        # 样本被正确分类且间隔足够：仅累计正则化梯度
                        dw += w

                # 计算平均梯度，批量更新参数
                dw /= n_samples
                db /= n_samples
                w -= self.learning_rate * dw
                b -= self.learning_rate * db

                # 每1000轮打印一次损失，监控收敛
                if self.verbose and (epoch + 1) % 1000 == 0:
                    current_loss = self._hinge_loss(X, y_binary, w, b)
                    print(f"Epoch {epoch+1}/{self.epochs} | 总损失: {current_loss:.4f}")

            # 存储当前类别的模型参数
            self.models[c] = (w, b)
            if self.verbose:
                print(f"类别 {c} 训练完成，最终权重w: {np.round(w,4)}, 偏置b: {np.round(b,4)}")

    def predict(self, X):
        """模型预测：选择决策函数值最大的类别"""
        predictions = []
        for x in X:
            class_scores = {}
            # 计算样本在每个二分类器上的得分
            for c, (w, b) in self.models.items():
                score = np.dot(w, x) + b
                class_scores[c] = score
            # 选择得分最高的类别作为最终预测
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)
        return np.array(predictions)

# ---------------------- 6. K-means聚类算法核心类 ----------------------
class KMeans:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, n_init=10):
        self.n_clusters = n_clusters  # 聚类簇数K，对应鸢尾花3个类别
        self.max_iters = max_iters    # 最大迭代次数
        self.tol = tol                # 收敛阈值
        self.n_init = n_init          # 多次初始化，选最优结果
        self.centroids = None          # 最终最优簇中心
        self.labels = None             # 最终最优簇标签
        self.best_wcss = float('inf')  # 最优模型的簇内平方和

    def _calculate_wcss(self, X, labels, centroids):
        """计算簇内平方和WCSS，用于选择最优初始化结果"""
        wcss = 0
        for k in range(self.n_clusters):
            cluster_samples = X[labels == k]
            wcss += np.sum(np.linalg.norm(cluster_samples - centroids[k], axis=1) ** 2)
        return wcss

    def _init_centroids_kmeans_plus_plus(self, X):
        """K-means++初始化簇中心，确保初始点分散"""
        n_samples, n_features = X.shape
        centroids = []
        # 1. 随机选择第一个簇中心
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])
        # 2. 依次选择剩余K-1个簇中心
        for _ in range(1, self.n_clusters):
            # 计算每个样本到已选簇中心的最小距离平方
            distances = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centroids)
                for x in X
            ])
            # 按距离平方的概率分布选择下一个中心
            probs = distances / distances.sum()
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def _fit_single_run(self, X):
        """单次K-means训练流程"""
        n_samples, n_features = X.shape
        # K-means++初始化
        centroids = self._init_centroids_kmeans_plus_plus(X)
        labels = None

        for i in range(self.max_iters):
            # 簇分配
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)
            # 簇中心更新
            new_centroids = np.array([
                X[new_labels == k].mean(axis=0)
                for k in range(self.n_clusters)
            ])
            # 收敛判断
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels = new_labels
            if centroid_shift < self.tol:
                break
        # 计算本次训练的WCSS
        wcss = self._calculate_wcss(X, labels, centroids)
        return labels, centroids, wcss

    def fit(self, X):
        """多次初始化，选择WCSS最小的最优模型"""
        print(f"开始K-means训练，共执行{self.n_init}次初始化，选择最优结果...")
        for run_idx in range(self.n_init):
            run_labels, run_centroids, run_wcss = self._fit_single_run(X)
            # 更新最优模型
            if run_wcss < self.best_wcss:
                self.best_wcss = run_wcss
                self.labels = run_labels
                self.centroids = run_centroids
            print(f"第{run_idx+1}次初始化完成 | 簇内平方和WCSS: {run_wcss:.4f}")
        print(f"\n最优模型已选定 | 最小WCSS: {self.best_wcss:.4f}")

    def predict(self, X):
        """预测新样本的簇标签"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# ---------------------- 7. 标签映射函数 ----------------------
def map_cluster_labels_to_true(y_true, y_cluster):
    """将聚类标签映射到与真实标签最匹配的编号"""
    from scipy.stats import mode
    label_mapping = {}
    for cluster_id in np.unique(y_cluster):
        true_label = mode(y_true[y_cluster == cluster_id], keepdims=False).mode
        label_mapping[cluster_id] = true_label
    mapped_labels = np.array([label_mapping[cluster_id] for cluster_id in y_cluster])
    return mapped_labels, label_mapping

# ---------------------- 8. 鸢尾花实验主流程 ----------------------
if __name__ == "__main__":
    # 8.1 数据加载与探索
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    print("="*60)
    print("鸢尾花数据集基本信息")
    print(f"样本总数：{X.shape[0]} | 特征数：{X.shape[1]}")
    print(f"类别数：{len(class_names)} | 类别名称：{class_names}")
    print(f"特征名称：{feature_names}")
    print("="*60)

    # 8.2 算法选择（交互式选项）
    print("\n请选择算法类型：")
    print("1. 决策树（监督学习分类，基于基尼系数）")
    print("2. 支持向量机（监督学习分类，优化版线性SVM）")
    print("3. K-means聚类（无监督学习，含PCA降维可视化）")
    choice = input("请输入选项编号（1、2或3）：").strip()

    # 通用评估函数
    def calc_confusion_matrix(y_true, y_pred, n_classes=3):
        """计算混淆矩阵"""
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label][pred_label] += 1
        return cm

    # 8.3 根据选择执行对应算法
    if choice == "1":
        print("\n已选择：决策树分类器")
        print("="*60)
        # 数据划分（70%训练，30%测试）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"训练集样本数：{X_train.shape[0]} | 测试集样本数：{X_test.shape[0]}")
        print("="*60)

        # 模型训练与预测
        model = DecisionTreeClassifier(min_samples_split=2, max_depth=3)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 模型评估
        print("\n" + "="*60)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"训练集准确率：{train_acc:.4f}")
        print(f"测试集准确率：{test_acc:.4f}")
        print("="*60)

        cm = calc_confusion_matrix(y_test, y_test_pred)
        print("测试集混淆矩阵")
        print(f"行：真实类别 {class_names}")
        print(f"列：预测类别 {class_names}")
        print(cm)
        print("="*60)

        # 决策树可视化
        print("\n正在生成决策树分类结果可视化...")
        X_all = np.vstack((X_train, X_test))
        y_all = np.hstack((y_train, y_test))
        plot_classification_results(model, X_all, y_all, class_names, "决策树分类器")
        print("可视化图表已生成！")
        print("="*60)

    elif choice == "2":
        print("\n已选择：优化版支持向量机分类器")
        print("="*60)
        # 数据划分（70%训练，30%测试）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"训练集样本数：{X_train.shape[0]} | 测试集样本数：{X_test.shape[0]}")
        print("="*60)

        # SVM特征标准化（Z-score，仅用训练集统计量，避免数据泄露）
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0) + 1e-8
        X_train_scaled = (X_train - X_train_mean) / X_train_std
        X_test_scaled = (X_test - X_train_mean) / X_train_std
        print("已完成特征标准化（Z-score）")
        print("="*60)

        # 模型训练与预测
        model = LinearSVM(learning_rate=0.001, C=20.0, epochs=10000, verbose=True)
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # 模型评估
        print("\n" + "="*60)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"训练集准确率：{train_acc:.4f}")
        print(f"测试集准确率：{test_acc:.4f}")
        print("="*60)

        cm = calc_confusion_matrix(y_test, y_test_pred)
        print("测试集混淆矩阵")
        print(f"行：真实类别 {class_names}")
        print(f"列：预测类别 {class_names}")
        print(cm)
        print("="*60)

        # SVM可视化
        print("\n正在生成SVM分类结果可视化...")
        X_all = np.vstack((X_train, X_test))
        y_all = np.hstack((y_train, y_test))
        plot_classification_results(model, X_all, y_all, class_names, "线性SVM分类器",
                                   scaler_mean=X_train_mean, scaler_std=X_train_std)
        print("可视化图表已生成！")
        print("="*60)

    elif choice == "3":
        print("\n已选择：K-means聚类算法（含优化可视化）")
        print("="*60)
        # K-means特征标准化（Z-score）
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std
        print("已完成特征标准化（Z-score）")
        print("="*60)

        # 模型训练
        model = KMeans(n_clusters=3, max_iters=1000, tol=1e-5, n_init=20)
        model.fit(X_scaled)
        y_cluster = model.labels

        # 标签映射，让聚类标签与真实标签对齐
        y_mapped, label_mapping = map_cluster_labels_to_true(y, y_cluster)
        print(f"\n簇标签-真实标签映射关系：{label_mapping}")

        # 模型评估
        print("\n" + "="*60)
        silhouette_avg = silhouette_score(X_scaled, y_cluster)
        print(f"轮廓系数（簇质量评估）：{silhouette_avg:.4f}")
        ari = adjusted_rand_score(y, y_cluster)
        print(f"调整兰德指数（与真实标签一致性）：{ari:.4f}")
        mapped_acc = accuracy_score(y, y_mapped)
        print(f"标签映射后准确率：{mapped_acc:.4f}")
        print("="*60)

        # 输出最终簇中心
        print("\n最终簇中心坐标（标准化后）：")
        for k in range(model.n_clusters):
            print(f"簇 {k} 中心：{np.round(model.centroids[k], 4)}")
        print("="*60)

        # K-means可视化
        print("\n正在生成K-means聚类结果可视化...")
        # PCA降维
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(model.centroids)

        # 中文显示设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 双图对比可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        cmap = 'viridis'

        # --- 左图：映射后的K-means聚类结果 ---
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_mapped, cmap=cmap, s=50, alpha=0.8, edgecolors='k')
        ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='x', color='black', s=150, linewidths=3, label='簇中心')
        ax1.set_title('K-means聚类结果（标签映射后）', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PCA主成分1（方差解释率：{pca.explained_variance_ratio_[0]:.2%}）', fontsize=12)
        ax1.set_ylabel(f'PCA主成分2（方差解释率：{pca.explained_variance_ratio_[1]:.2%}）', fontsize=12)
        ax1.legend()
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='聚类类别', ticks=[0,1,2])
        cbar1.ax.set_yticklabels(class_names)

        # --- 右图：真实标签分布 ---
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, s=50, alpha=0.8, edgecolors='k')
        ax2.set_title('鸢尾花真实标签分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PCA主成分1（方差解释率：{pca.explained_variance_ratio_[0]:.2%}）', fontsize=12)
        ax2.set_ylabel(f'PCA主成分2（方差解释率：{pca.explained_variance_ratio_[1]:.2%}）', fontsize=12)
        cbar2 = plt.colorbar(scatter2, ax=ax2, label='真实类别', ticks=[0,1,2])
        cbar2.ax.set_yticklabels(class_names)

        # 调整布局并显示
        plt.tight_layout()
        plt.show()
        print("可视化图表已生成！")
        print("="*60)

    else:
        print("无效选项，请重新运行程序并输入1、2或3。")
        exit()
