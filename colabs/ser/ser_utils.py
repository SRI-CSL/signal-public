import numpy as np


def depth_vtree(tree, node):
    p, _, _ = extract_rule_vtree(tree, node)
    return len(p)


def find_parent_vtree(tree, i_node):
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:

        try:
            p = list(tree.children_left).index(i_node)
            b = -1
        except:
            p = p
        try:
            p = list(tree.children_right).index(i_node)
            b = 1
        except:
            p = p

    return p, b


def extract_rule_vtree(tree, node):

    feats = list()
    ths = list()
    bools = list()
    nodes = list()
    b = 1
    if node != 0:
        while b != 0:

            feats.append(tree.feature[node])
            ths.append(tree.threshold[node])
            bools.append(b)
            nodes.append(node)
            node, b = find_parent_vtree(tree, node)

        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)

    return np.array(feats), np.array(ths), np.array(bools)


def depth_rf(rf):
    d = 0
    for p in rf.estimators_:
        d = d + p.tree_.max_depth
    return d / len(rf.estimators_)


def depth(dec_tree, node):
    p, _, _ = extract_rule(dec_tree, node)
    return len(p)


def depth_array(dec_tree, indices):
    depths = np.zeros(np.array(indices).size)
    for i, _ in enumerate(indices):
        depths[i] = depth(dec_tree, i)
    return depths


def leaf_error(tree, node):
    if np.sum(tree.value[node]) == 0:
        return 0
    else:
        return 1 - np.max(tree.value[node]) / np.sum(tree.value[node])


def subtree_error(tree, node):
    if node == -1:
        return 0
    else:

        if tree.feature[node] == -2:
            return leaf_error(tree, node)
        else:
            # Not a leaf

            nr = np.sum(tree.value[tree.children_right[node]])
            nl = np.sum(tree.value[tree.children_left[node]])

            if nr + nl == 0:
                return 0
            else:
                er = subtree_error(tree, tree.children_right[node])
                el = subtree_error(tree, tree.children_left[node])

                return (el * nl + er * nr) / (nl + nr)


def bootstrap(size):
    return np.random.choice(np.linspace(0, size - 1, size).astype(int), size, replace=True)


def get_node_distribution(dec_tree, node_index):
    tree = dec_tree.tree_
    Q = tree.value[node_index]
    return np.asarray(Q)

def extract_rule(dec_tree, i_node):
    return extract_rule_vtree(dec_tree.tree_, i_node)


def extract_leaves_rules(dec_tree):
    leaves = np.where(dec_tree.tree_.feature == -2)[0]

    rules = np.zeros(leaves.size, dtype=object)
    for k, f in enumerate(leaves):
        rules[k] = extract_rule(dec_tree, f)

    return leaves, rules


def find_parent(dec_tree, i_node):
    return find_parent_vtree(dec_tree.tree_, i_node)

def sub_nodes(tree, node):
    if (node == -1):
        return list()
    if (tree.feature[node] == -2):
        return [node]
    else:
        return [node] + sub_nodes(tree, tree.children_left[node]) + sub_nodes(tree, tree.children_right[node])


def fuse_trees(tree1, leaf_f, tree2):
    """adding tree tree2 to leaf f of tree tree1"""

    dic = tree1.__getstate__().copy()
    dic2 = tree2.__getstate__().copy()

    size_init = tree1.node_count

    if depth_vtree(tree1, leaf_f) + dic2['max_depth'] > dic['max_depth']:
        dic['max_depth'] = depth_vtree(tree1, leaf_f) + tree2.max_depth

    dic['capacity'] = tree1.capacity + tree2.capacity - 1
    dic['node_count'] = tree1.node_count + tree2.node_count - 1

    dic['nodes'][leaf_f] = dic2['nodes'][0]

    if (dic2['nodes']['left_child'][0] != - 1):
        dic['nodes']['left_child'][leaf_f] = dic2[
            'nodes']['left_child'][0] + size_init - 1
    else:
        dic['nodes']['left_child'][leaf_f] = -1
    if (dic2['nodes']['right_child'][0] != - 1):
        dic['nodes']['right_child'][leaf_f] = dic2[
            'nodes']['right_child'][0] + size_init - 1
    else:
        dic['nodes']['right_child'][leaf_f] = -1

    # Attention vector impurity not updated

    dic['nodes'] = np.concatenate((dic['nodes'], dic2['nodes'][1:]))
    dic['nodes']['left_child'][size_init:] = (dic['nodes']['left_child'][
                                              size_init:] != -1) * (dic['nodes']['left_child'][size_init:] + size_init) - 1
    dic['nodes']['right_child'][size_init:] = (dic['nodes']['right_child'][
                                               size_init:] != -1) * (dic['nodes']['right_child'][size_init:] + size_init) - 1

    values = np.concatenate((dic['values'], np.zeros((dic2['values'].shape[
                            0] - 1, dic['values'].shape[1], dic['values'].shape[2]))), axis=0)

    dic['values'] = values

    # Attention :: (potentially important)
    (Tree, (n_f, n_c, n_o), b) = tree1.__reduce__()
    #del tree1
    #del tree2

    tree1 = Tree(n_f, n_c, n_o)

    tree1.__setstate__(dic)
    return tree1


def fuse_dec_trees(dec_tree_1, f, dec_tree_2):
    """adding tree dec_tree_2 to leaf f of tree dec_tree_1"""
    # dec_tree = sklearn.tree.DecisionTreeClassifier()
    size_init = dec_tree_1.tree_.node_count
    dec_tree_1.tree_ = fuse_trees(dec_tree_1.tree_, f, dec_tree_2.tree_)

    try:
        dec_tree_1.tree_.value[size_init:, :, dec_tree_2.classes_.astype(int)] = dec_tree_2.tree_.value[1:, :, :]
    except IndexError as e:
        print("IndexError : size init : ", size_init,
              "\ndTree2.classes_ : ", dec_tree_2.classes_)
        print(e)
    dec_tree_1.max_depth = dec_tree_1.tree_.max_depth
    return dec_tree_1


def cut_from_left_right(dec_tree, node, bool_left_right):
    dic = dec_tree.tree_.__getstate__().copy()

    node_to_rem = list()
    size_init = dec_tree.tree_.node_count

    p, b = find_parent(dec_tree, node)

    if bool_left_right == 1:
        repl_node = dec_tree.tree_.children_left[node]
        # node_to_rem = [node] + sub_nodes(dTree.tree_,dTree.tree_.children_right[node])
        node_to_rem = [node, dec_tree.tree_.children_right[node]]
    elif bool_left_right == -1:
        repl_node = dec_tree.tree_.children_right[node]
        # node_to_rem = [node] + sub_nodes(dTree.tree_,dTree.tree_.children_left[node])
        node_to_rem = [node, dec_tree.tree_.children_left[node]]

    indices = list(set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))

    dic['capacity'] = dec_tree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dec_tree.tree_.node_count - len(node_to_rem)

    if b == 1:
        dic['nodes']['right_child'][p] = repl_node
    elif b == -1:
        dic['nodes']['left_child'][p] = repl_node

    # new_size = len(ind)
    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']
    # print('width before: ', dic['nodes'].size)
    dic['nodes'] = dic['nodes'][indices]
    dic['values'] = dic['values'][indices]
    #print('width after:',dic['nodes'].size)

    for i, new in enumerate(indices):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = indices.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = indices.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dec_tree.tree_.__reduce__()
    del dec_tree.tree_

    dec_tree.tree_ = Tree(n_f, n_c, n_o)
    dec_tree.tree_.__setstate__(dic)
    depths = depth_array(dec_tree, np.linspace(0, dec_tree.tree_.node_count - 1, dec_tree.tree_.node_count).astype(int))
    dec_tree.tree_.max_depth = np.max(depths)

    return indices.index(repl_node)


def cut_into_leaf2(dec_tree, node):
    dic = dec_tree.tree_.__getstate__().copy()

    node_to_rem = list()
    size_init = dec_tree.tree_.node_count

    node_to_rem = node_to_rem + sub_nodes(dec_tree.tree_, node)[1:]
    node_to_rem = list(set(node_to_rem))

    indices = list(set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))
    depths = depth_array(dec_tree, indices)
    dic['max_depth'] = np.max(depths)

    dic['capacity'] = dec_tree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dec_tree.tree_.node_count - len(node_to_rem)

    dic['nodes']['feature'][node] = -2
    dic['nodes']['left_child'][node] = -1
    dic['nodes']['right_child'][node] = -1

    # new_size = len(ind)
    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']
    dic['nodes'] = dic['nodes'][indices]
    dic['values'] = dic['values'][indices]

    for i, new in enumerate(indices):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = indices.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = indices.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dec_tree.tree_.__reduce__()
    del dec_tree.tree_

    dec_tree.tree_ = Tree(n_f, n_c, n_o)
    dec_tree.tree_.__setstate__(dic)

    return indices.index(node)


def add_to_parents(dec_tree, node, values):

    p, b = find_parent(dec_tree, node)

    if b != 0:
        dec_tree.tree_.value[p] = dec_tree.tree_.value[p] + values
        add_to_parents(dec_tree, p, values)


def add_to_child(dec_tree, node, values):

    l = dec_tree.tree_.children_left[node]
    r = dec_tree.tree_.children_right[node]

    if r != -1:
        dec_tree.tree_.value[r] = dec_tree.tree_.value[r] + values
        add_to_child(dec_tree, r, values)
    if l != -1:
        dec_tree.tree_.value[l] = dec_tree.tree_.value[l] + values
        add_to_child(dec_tree, l, values)

