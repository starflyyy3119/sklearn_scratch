class Node(object):
    def __init__(self, feature=None, feature_val=None, partitions=None, depth=None,
                 is_leaf=False, leaf_val=None, is_continuous=False, classes=None):
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
        self.classes = classes

        # 用于剪枝的特征
        self.cost_with_subtree = None  # 以当前 node 为根节点的子树的损失(CART 分类数用 gini index 度量)
        self.cost_as_leaf = None  # 以当前 node 为 单节点树的损失
        self.nleaves = None

    def change__to_leaf(self):
        self.left = self.right = None
        self.feature = None
        self.feature_val = None
        self.partitions = None
        self.is_continuous = None

        self.is_leaf = True
        self.leaf_val = sorted(self.classes.items(), key=lambda x: x[1], reverse=True)[0][0]


