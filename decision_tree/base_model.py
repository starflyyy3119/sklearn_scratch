from decision_tree_v2 import *
import pandas as pd
# 用 kaggle 上的 titanic 数据集做实验，实现 base_model, 目前没有实现缺失值处理，所以将有缺失值的列都去掉
# base model

df = pd.read_csv('./data/train.csv')

# 查看各个变量中 na 的占比
print(df.isna().sum() / len(df) * 100)  # 发现 cabin 77.10%, age 19.86%, embarked 0.22% 有缺失, Ticket 是票编号，没啥用

df_test = pd.read_csv('./data/test.csv')

print(df_test.isna().sum() / len(df) * 100)  # 发现 fare 0.11 %, cabin 36.7%, age 9.65% 三列有缺失

# base model: 将有缺失的列都去除

base_train = df.iloc[:, [2, 4, 6, 7, 1]]

print(base_train.isna().sum() / len(df) * 100)

base_test = df_test.iloc[:, [1, 3, 5, 6]]

print(base_test.isna().sum() / len(base_test) * 100)

tree = DecisionTree(min_leaf_size=10, max_depth=5)

print(cross_validation_score(tree, base_train))  # 78.42%

test_predict = tree.fit(base_train).predict(base_test)

# test_predict.to_csv('./data/result.csv')  # base_model 在榜 0.76555

tree.print_tree()