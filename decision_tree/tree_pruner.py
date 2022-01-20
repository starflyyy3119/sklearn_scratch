import numpy as np
import copy
import pandas as pd


class TreePruner(object):
    def __init__(self, tree, data):
        # 用于剪枝的变量
        self.tree = tree
        self.data = data

        # 用于剪枝的参数
        self.alpha = 0x3f3f3f3f
        self.best_trees = {}
        self.best_tree = None
        self.acc = {}

        # 先将原始的树加入树字典
        self.best_trees[0] = copy.deepcopy(self.tree)

    # 刘建平的博客: https://www.cnblogs.com/pinard/p/6053344.html
    def post_prune(self):
        self.alpha = 0x3f3f3f3f

        self.__cal__(self.tree)  # 从下到上计算相应的参数

        self.__prune__(self.tree)  # 从上到下进行剪枝

        self.best_trees[self.alpha] = copy.deepcopy(self.tree)  # 将剪枝完成的树加入候选集合

        if self.tree.left is not None or self.tree.right is not None:
            self.post_prune()
        else:
            # 使用cross validation 选择最好的树
            for alpha, tree in self.best_trees.items():
                score = self.cross_validation_score(tree, self.data)
                self.acc[alpha] = score

            best_alpha = sorted(self.acc.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
            self.best_tree = self.best_trees[best_alpha]

    def get_best_tree(self):
        return self.best_tree

    def __prune__(self, node):
        # 按照根左右的遍历方式
        if node.is_leaf:
            return
        else:
            now_alpha = (node.cost_as_leaf - node.cost_with_subtree) / (node.nleaves - 1)
            if now_alpha <= self.alpha:
                node.change__to_leaf()
            else:
                self.__prune__(node.left)
                self.__prune__(node.right)

    def __cal__(self, node):
        if node.is_leaf:
            node.cost_with_subtree = node.cost_as_leaf = self.__gini__(node.classes)
            node.nleaves = 1
        else:
            # 采用左右根的二叉树遍历方式
            self.__cal__(node.left)
            self.__cal__(node.right)

            node.nleaves = node.left.nleaves + node.right.nleaves
            node.cost_as_leaf = self.__gini__(node.classes)

            left, right = len(node.partitions['left']), len(node.partitions['right'])
            node.cost_with_subtree = left / (left + right) * node.left.cost_with_subtree + \
                                     right / (left + right) * node.right.cost_with_subtree

            self.alpha = min(self.alpha,
                             (node.cost_as_leaf - node.cost_with_subtree) / (node.nleaves - 1))

    def __gini__(self, classes):
        l = np.array([value for _, value in classes.items()])
        p = l / np.sum(l)
        return 1 - np.sum(p ** 2)

    def __predict_row__(self, x, node):
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

    def __predict__(self, X, node):
        return X.apply(lambda x: self.__predict_row__(x, node), axis=1)  # apply functions to each row

    def cross_validation_score(self, node, dataset, k_folds=5):
        accuracy = list()
        interval = len(dataset) / k_folds
        for i in range(k_folds):
            # 删除 start : end 间的行
            start, end = int(interval * i), int(min(interval * (i + 1), len(dataset) - 1))
            train = pd.concat([dataset.iloc[: start, :], dataset.iloc[end:, :]])
            test = dataset.iloc[start: end, :]
            predicted = self.__predict__(test, node)
            score = accuracy_metric(predicted, test.iloc[:, -1])
            accuracy.append(score)
        return sum(accuracy) / len(accuracy)


def accuracy_metric(predicted, actual):
    return np.sum(predicted == actual) / len(actual) * 100
