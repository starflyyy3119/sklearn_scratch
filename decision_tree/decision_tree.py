from random import seed
from random import randrange
from csv import reader


def gini(d, classes):
    """
    计算数据集 D 的基尼系数
    :param d: 数据集
    :param classes: 类别 [0, 1]
    :return: 1 - sum((|C_k| / |D|) ** 2), ｜C_k｜ 是第 k 类的样本数目
    """
    # 防止数据集中没有数据
    if len(d) == 0:
        return 0.0

    score = 0.0
    for class_val in classes:
        c_k = sum([row[-1] == class_val for row in d])
        score += (c_k / len(d)) ** 2
    return 1.0 - score


def gini_index(groups, classes):
    """
    根据 A = a，将 D 分成 D1 和 D2 两部分后的基尼系数
    :param groups: 数据集 D 利用 A = a 进行分割后产生的 D1 和 D2
                   [[[1, 1], [1, 0]], [[1, 1], [1, 0]]]
    :param classes: 类别 [0, 1]
    :return:
    """
    len_of_d1 = len(groups[0])
    len_of_d2 = len(groups[1])
    len_of_d = len_of_d1 + len_of_d2
    return len_of_d1 / len_of_d * gini(groups[0], classes) + \
        len_of_d2 / len_of_d * gini(groups[1], classes)


# test Gini values
# print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
# print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

def test_split(index, value, dataset):
    """
    根据 A < a 和 A >= a 将数据集分成两部分
    :param index: 特征的序号
    :param value: 特征的值
    :param dataset: 数据集
    :return: left, right
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_split(dataset):
    """
    得到数据集的当前最优划分, 只考虑了连续变量
    :param dataset: 数据集
    :return: 当前最优划分
    """
    class_values = list(set(row[-1] for row in dataset))  # 类别名
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini_val = gini_index(groups, class_values)
            if gini_val < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini_val, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


#
#
# # dataset = [[2.771244718, 1.784783929, 0],
# #            [1.728571309, 1.169761413, 0],
# #            [3.678319846, 2.81281357, 0],
# #            [3.961043357, 2.61995032, 0],
# #            [2.999208922, 2.209014212, 0],
# #            [7.497545867, 3.162953546, 1],
# #            [9.00220326, 3.339047188, 1],
# #            [7.444542326, 0.476683375, 1],
# #            [10.12493903, 3.234550982, 1],
# #            [6.642287351, 3.319983761, 1]]
# # split = get_split(dataset)
# # print('Split: [X%d < %.3f]' % ((split['index'] + 1), split['value']))


def to_terminal(group):
    """
    产生叶子节点
    :param group: 被分为一组的样本
    :return: 返回该组中大多数样本所属的类
    """
    res = dict()
    for row in group:
        key = row[-1]
        tmp = res.get(key, 0)
        res[key] = tmp + 1
    b_k, b_v = -1, -1
    for key, value in res.items():
        if value > b_v:
            b_k = key
            b_v = value
    return b_k


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
