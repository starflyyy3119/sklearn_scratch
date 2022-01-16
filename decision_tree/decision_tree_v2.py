import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


# reference 基础理论: 《统计学习方法》
# 树模型的构建: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# 类的实现: https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/
# 缺失值的处理(训练及测试中都可能出现): https://blog.csdn.net/u012328159/article/details/79413610


class Node(object):
    def __init__(self, feature=None, feature_val=None, partitions=None, depth=None,
                 is_leaf=False, leaf_val=None, is_continuous=False):
        # 非叶子节点
        self.left = None
        self.right = None
        self.feature = feature
        self.feature_val = feature_val
        self.partitions = partitions
        self.is_continuous = is_continuous

        # 叶子节点
        self.is_leaf = is_leaf
        self.leaf_val = leaf_val

        # 当前节点的深度，共有
        self.depth = depth


class DecisionTree(object):
    def __init__(self, max_depth=10, min_leaf_size=5, criterion='gini', pruning=None):
        """
        :param max_depth: 最大深度，default 10
        :param min_leaf_size: 叶子节点最小样本数目, default 5
        :param criterion: 'gini', 'infogain', 'gainratio'
        :param pruning: 'pre_pruning', 'post_pruning'
        """
        assert criterion in ('gini', 'infogain', 'gainratio')
        assert pruning in (None, 'pre_pruning', 'post_pruning')
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.criterion = criterion
        self.pruning = pruning

        # 建立的决策树的根节点
        self.root = None

    def fit(self, data):
        self.root = self.get_node(data, 1)
        self.split(self.root)
        return self

    def split(self, node):
        left, right, depth = node.partitions['left'], node.partitions['right'], node.depth
        del node.partitions

        if len(left) == 0 or len(right) == 0:
            node.left = node.right = self.to_leaf(pd.concat([left, right]), depth + 1)
            return
        if node.depth >= self.max_depth:
            node.left, node.right = self.to_leaf(left, depth + 1), self.to_leaf(right, depth + 1)
            return
        if len(left) <= self.min_leaf_size:
            node.left = self.to_leaf(left, depth + 1)
        else:
            node.left = self.get_node(left, depth + 1)
            self.split(node.left)
        if len(right) <= self.min_leaf_size:
            node.right = self.to_leaf(right, depth + 1)
        else:
            node.right = self.get_node(right, depth + 1)
            self.split(node.right)

    def to_leaf(self, partition, depth):
        """
        产生叶子节点
        :param depth: 当前叶子节点的深度
        :param partition: 当前分到该节点的数据
        :return: 当前 partition 中数目最多的类别
        """
        return Node(is_leaf=True, leaf_val=partition.iloc[:, -1].value_counts().index[0], depth=depth)

    def get_node(self, data, depth):
        """
        得到数据的最优划分，并返回一个 Node 对象
        :param data: 数据
        :param depth: 当前节点的深度
        :return: Node 对象
        """
        if self.criterion == 'gini':
            return self.get_node_by_gini(data, depth)
        elif self.criterion == 'infogain':
            return self.get_node_by_infogain(data, depth)
        else:
            return self.get_node_by_gainratio(data, depth)

    def get_node_by_gini(self, data, depth):
        """
        根据 gini 系数得到最优划分(feature, feature_val, partitions)
        :param data: 当前数据, DataFrame 对象
        :param depth: 当前节点的深度
        :return: Node 对象
        """
        best_feature, best_value, best_score, best_partitions, best_is_continuous = 999, 999, 999, None, False
        # 有列号取的是列号, 没有列号取的是序号, 从 0 开始
        for feature in data.columns[:len(data.columns) - 1]:
            is_continuous = type_of_target(data[feature]) == 'continuous'
            # 得到该 feature 的所有值
            feature_vals = data[feature].unique()
            for feature_val in feature_vals:
                partitions = self.get_binary_partitions(feature, feature_val, data, is_continuous)
                gini_val = self.gini_index(partitions)
                if gini_val < best_score:
                    best_feature, best_value, best_score, best_partitions, best_is_continuous = \
                        feature, feature_val, gini_val, partitions, is_continuous
        return Node(best_feature, best_value, best_partitions, depth, is_continuous=best_is_continuous)

    def get_binary_partitions(self, feature, feature_val, data, is_continuous=False):
        """
        得到数据集的按照 feature 及对应 feature_val 的二叉分割， CART 算法是二叉树
        :param feature: 特征
        :param feature_val: 该特征对应的值
        :param data: 当前数据
        :param is_continuous: 标识该特征是否是连续型变量
        :return:
        """
        # 连续型变量
        if is_continuous:
            left = data.loc[data[feature] < feature_val]
            right = data.loc[data[feature] >= feature_val]
        # 离散型变量
        else:
            left = data.loc[data[feature] == feature_val]
            right = data.loc[data[feature] != feature_val]
        return {'left': left, 'right': right}

    def gini_index(self, partitions):
        """
        计算一个给定划分的 gini 系数, P70 公式 5.25
        :param partitions: 给定的划分 {1: left, 2: right}, left 和 right 都是 DataFrame
        :return: gini 系数
        """
        left, right = partitions['left'], partitions['right']
        total_rows = len(left) + len(right)
        return len(left) / total_rows * self.gini(left.iloc[:, -1]) + \
               len(right) / total_rows * self.gini(right.iloc[:, -1])

    def gini(self, y):
        p = pd.value_counts(y) / y.shape[0]
        gini_val = 1 - np.sum(p ** 2)
        return gini_val

    def print_tree(self, node=None):
        if node is None:
            node = self.root
        # 非叶子节点
        if not node.is_leaf:
            if node.is_continuous:
                print('%s[%s < %.3f]' % (node.depth * ' ', node.feature, node.feature_val))
            else:
                print('%s[%s = %s]' % (node.depth * ' ', node.feature, node.feature_val))
            self.print_tree(node.left)
            self.print_tree(node.right)

        # 叶子节点
        else:
            print('%s[%s]' % (node.depth * ' ', node.leaf_val))

    def __predict_row__(self, x, node=None):
        if node is None:
            node = self.root

        # 叶子节点
        if node.is_leaf:
            return node.leaf_val

        # 非叶子节点, 且为连续变量
        if node.is_continuous:
            if x[node.feature] < node.feature_val:
                return self.__predict_row__(x, node.left)
            else:
                return self.__predict_row__(x, node.right)
        # 非叶子节点，且为离散变量
        else:
            if x[node.feature] == node.feature_val:
                return self.__predict_row__(x, node.left)
            else:
                return self.__predict_row__(x, node.right)

    def predict(self, X):
        return X.apply(lambda x: self.__predict_row__(x), axis=1)  # apply functions to each row


def accuracy_metric(predicted, actual):
    return np.sum(predicted == actual) / len(actual) * 100


def cross_validation_score(model, dataset, k_folds=5):
    accuracy = list()
    interval = len(dataset) / k_folds
    for i in range(k_folds):
        # 删除 start : end 间的行
        start, end = int(interval * i), int(min(interval * (i + 1), len(dataset) - 1))
        train = pd.concat([dataset.iloc[: start, :], dataset.iloc[end:, :]])
        test = dataset.iloc[start: end, :]
        predicted = model.fit(train).predict(test.iloc[:, : len(test) - 1])
        score = accuracy_metric(predicted, test.iloc[:, -1])
        accuracy.append(score)
    return sum(accuracy) / len(accuracy)

# 测试案例
if __name__ == '__main__':
    dataset = pd.read_csv('./data/data_banknote_authentication.csv')
    dataset.columns = ['X0', 'X1', 'X2', 'X3', 'y']
    tree = DecisionTree(min_leaf_size=10, max_depth=5)
    avg_acc = cross_validation_score(tree, dataset)
    print('The average accuracy of the model is %.2f' % avg_acc)