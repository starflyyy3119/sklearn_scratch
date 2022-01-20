# sklearn_scratch

## Decision Tree
- 实现了 [Cart 分类树生成算法](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/decision_tree_v2.py)
- 实现了 [Cart 分类树后剪枝算法](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/tree_pruner.py)
- 利用 graphviz 实现了决策树的可视化，简略版和详细版(详细版可协助理解剪枝过程):
 
*简略版未后剪枝*:

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/tree_easy_not_prune.png)

```python
# train 是训练集，要求 y 必须在最后一列
tree = DecisionTree(min_leaf_size=10, max_depth=3).fit(train)
tree.__cal__()
tree.to_dot_file('./dot_file/tree_easy_not_prune.dot', is_whole = False)
(graph,) = pydot.graph_from_dot_file('./dot_file/tree_easy_not_prune.dot')
graph.write_png('./fig/tree_easy_not_prune.png')
```


*简略版后剪枝*:

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/tree_easy.png)

```python
# train 是训练集，要求 y 必须在最后一列
tree = DecisionTree(min_leaf_size=10, max_depth=3, pruning="post_pruning").fit(train)
tree.to_dot_file('./dot_file/tree_easy.dot', is_whole = False)
(graph,) = pydot.graph_from_dot_file('./dot_file/tree_easy.dot')
graph.write_png('./fig/tree_easy.png')
```

*详细版未后剪枝*:

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/tree_complex_not_prune.png)

```python
# train 是训练集，要求 y 必须在最后一列
tree = DecisionTree(min_leaf_size=10, max_depth=3).fit(train)
tree.__cal__()
tree.to_dot_file('./dot_file/tree_complex_not_prune.dot', is_whole = True)
(graph,) = pydot.graph_from_dot_file('./dot_file/tree_complex_not_prune.dot')
graph.write_png('./fig/tree_complex_not_prune.png')
```

*详细版后剪枝*:

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/tree_complex.png)

```python
# train 是训练集，要求 y 必须在最后一列
tree = DecisionTree(min_leaf_size=10, max_depth=3, pruning="post_pruning").fit(train)
tree.to_dot_file('./dot_file/tree_complex.dot', is_whole = True)
(graph,) = pydot.graph_from_dot_file('./dot_file/tree_complex.dot')
graph.write_png('./fig/tree_complex.png')
```


- 加入了 [统计学习方法 P51 数据集](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/lihang.ipynb)

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/lihang.png)

- 加入了 [Kaggle titanic 数据集](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/titanic_decision_tree_model_with_feature_engineering.ipynb)

*未后剪枝*:

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/my_tree.png)

*后剪枝后*:

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/my_tree_prune.png)

*排名*:

![](https://github.com/starflyyy3119/sklearn_scratch/blob/master/decision_tree/fig/rank.png)
