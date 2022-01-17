from decision_tree_v2 import *
import pandas as pd
# 用 kaggle 上的 titanic 数据集做实验，实现 base_model, 目前没有实现缺失值处理，所以将有缺失值的列都去掉
# test
df = pd.read_csv('./data/train.csv')

# 查看各个变量中 na 的占比
print(df.isna().sum() / len(df) * 100)  # 发现 cabin 77.10%, age 19.86%, embarked 0.22% 有缺失, Ticket 是票编号，没啥用

df_test = pd.read_csv('./data/test.csv')

print(df_test.isna().sum() / len(df) * 100)  # 发现 fare 0.11 %, cabin 36.7%, age 9.65% 三列有缺失

# advance model: 现在能够处理缺失的变量了

advance_train = df.iloc[:, [2, 4, 5, 6, 7, 9, 11, 1]]
#
print(advance_train.isna().sum() / len(df) * 100)

advance_test = df_test.iloc[:, [1, 3, 4, 5, 6, 8, 10]]

print(advance_test.isna().sum() / len(advance_test) * 100)


tree = DecisionTree(min_leaf_size=10, max_depth=5)

# print(cross_validation_score(tree, advance_train))  # 有 cabin 80.78, 无 cabin 80.79
#
test_predict = tree.fit(advance_train).predict(advance_test)

test_predict.to_csv('./data/advance.csv')  # base_model 在榜 0.77990 排名 3457

tree.print_tree()

