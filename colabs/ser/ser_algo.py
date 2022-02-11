"""
Implementation of
[1] N. Segev, M. Harel, S. Mannor, K. Crammer, and R. El-Yaniv, “Learn on  source,  refine  on  target:  a  model  transfer  learning  framework  with random  forests,” *IEEE  transactions  on  pattern  analysis  and  machine intelligence*, vol. 39, no. 9, pp. 1811–1824, 2017.
[[link]](https://ieeexplore.ieee.org/document/7592407)  

[2] L. Minvielle, M. Atiq, S. Peignier and M. Mougeot, "Transfer Learning on Decision Tree with Class Imbalance", *2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)*, 2019, pp. 1003-1010.
[[link]](https://ieeexplore.ieee.org/document/8995296)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import copy
import ser_utils as su

DEBUG = True

def SER(node, dec_tree, X_target_node, y_target_node, original_ser=True, no_red_on_cl=False,
        cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None, ext_cond=None, leaf_loss_quantify=False,
        leaf_loss_threshold=None, coeffs=None, root_source_values=None, n_k_min=None):

    # CARE : Deep copy of value
    old_values = dec_tree.tree_.value[node].copy()
    maj_class = np.argmax(dec_tree.tree_.value[node, :])

    if cl_no_red is None:
        old_size_cl_no_red = 0
    else:
        old_size_cl_no_red = np.sum(dec_tree.tree_.value[node][:, cl_no_red])

    if (leaf_loss_quantify and 
        (no_red_on_cl or no_ext_on_cl
         and dec_tree.tree_.feature[node] == -2)):

        if no_red_on_cl:
            cl = cl_no_red[0]
        else:
            cl = cl_no_ext[0]

        ps_rf = dec_tree.tree_.value[node, 0, :] / sum(dec_tree.tree_.value[node, 0, :])
        p1_in_l = dec_tree.tree_.value[node, 0, cl] / root_source_values[cl]
        cond1 = np.power(1 - p1_in_l, n_k_min) > leaf_loss_threshold
        cond2 = np.argmax(np.multiply(coeffs, ps_rf)) == cl

    ### VALUES UPDATE ###
    vals_to_update = np.zeros((dec_tree.n_outputs_, dec_tree.n_classes_))

    for i in range(dec_tree.n_classes_):
        vals_to_update[:, i] = list(y_target_node).count(i)

    dec_tree.tree_.value[node] = vals_to_update
    dec_tree.tree_.n_node_samples[node] = np.sum(vals_to_update)
    dec_tree.tree_.weighted_n_node_samples[node] = np.sum(vals_to_update)

    if dec_tree.tree_.feature[node] == -2:
        if original_ser:
            if y_target_node.size > 0 and len(set(list(y_target_node))) > 1:
                # the class changes automatically according to
                # target by the values ​​updates

                dec_tree_to_add = DecisionTreeClassifier()

                try:
                    dec_tree_to_add.min_impurity_decrease = 0
                except:
                    dec_tree_to_add.min_impurity_split = 0
                dec_tree_to_add.fit(X_target_node, y_target_node)
                su.fuse_dec_trees(dec_tree, node, dec_tree_to_add)

            if DEBUG:
                print(f"Processing node: {node}")
            return node, False

        else:
            bool_no_red = False
            cond_extension = False

            if y_target_node.size > 0:
                # Extension
                if not no_ext_on_cl:
                    dec_tree_to_add = DecisionTreeClassifier()
                    # to make a complete tree
                    try:
                        dec_tree_to_add.min_impurity_decrease = 0
                    except:
                        dec_tree_to_add.min_impurity_split = 0
                    dec_tree_to_add.fit(X_target_node, y_target_node)
                    su.fuse_dec_trees(dec_tree, node, dec_tree_to_add)
                else:
                    cond_maj = (maj_class not in cl_no_ext)
                    cond_sub_target = ext_cond and (
                        maj_class in y_target_node) and (maj_class in cl_no_ext)
                    cond_leaf_loss = leaf_loss_quantify and not (
                        cond1 and cond2)

                    cond_extension = cond_maj or cond_sub_target or cond_leaf_loss

                    if cond_extension:
                        dec_tree_to_add = DecisionTreeClassifier()
                        # to make a complete tree
                        try:
                            dec_tree_to_add.min_impurity_decrease = 0
                        except:
                            dec_tree_to_add.min_impurity_split = 0
                        dec_tree_to_add.fit(X_target_node, y_target_node)
                        su.fuse_dec_trees(dec_tree, node, dec_tree_to_add)
                    else:
                        # Complicated not to induce any inconsistency in the
                        # values ​​by leaving the sheets intact in this way. 
                        # That said, it has no impact on the decision tree
                        # that we want to obtain (it has one on the probability tree
                        dec_tree.tree_.value[node] = old_values
                        dec_tree.tree_.n_node_samples[node] = np.sum(old_values)
                        dec_tree.tree_.weighted_n_node_samples[
                            node] = np.sum(old_values)
                        su.add_to_parents(dec_tree, node, old_values)
                        if no_red_on_cl:
                            bool_no_red = True

            # no red protection with values
            if (no_red_on_cl and y_target_node.size == 0 
                and old_size_cl_no_red > 0 and maj_class in cl_no_red):

                if leaf_loss_quantify:
                    if cond1 and cond2:
                        dec_tree.tree_.value[node] = old_values
                        dec_tree.tree_.n_node_samples[node] = np.sum(old_values)
                        dec_tree.tree_.weighted_n_node_samples[node] = np.sum(old_values)
                        su.add_to_parents(dec_tree, node, old_values)
                        bool_no_red = True
                else:
                    dec_tree.tree_.value[node] = old_values
                    dec_tree.tree_.n_node_samples[node] = np.sum(old_values)
                    dec_tree.tree_.weighted_n_node_samples[node] = np.sum(old_values)
                    su.add_to_parents(dec_tree, node, old_values)
                    bool_no_red = True
            if DEBUG:
                print(f"Node: {node}, bool_no_red: {bool_no_red}")
            return node, bool_no_red

    ### Left / right target computation ###
    bool_test = X_target_node[:, dec_tree.tree_.feature[node]] <= dec_tree.tree_.threshold[node]
    not_bool_test = X_target_node[:, dec_tree.tree_.feature[node]] > dec_tree.tree_.threshold[node]

    ind_left = np.where(bool_test)[0]
    ind_right = np.where(not_bool_test)[0]

    X_target_node_left = X_target_node[ind_left]
    y_target_node_left = y_target_node[ind_left]

    X_target_node_right = X_target_node[ind_right]
    y_target_node_right = y_target_node[ind_right]

    if original_ser:
        new_node_left, bool_no_red_l = SER(dec_tree.tree_.children_left[node], dec_tree, X_target_node_left, y_target_node_left, original_ser=True)
        node, b = su.find_parent(dec_tree, new_node_left)

        new_node_right, bool_no_red_r = SER(dec_tree.tree_.children_right[node], dec_tree, X_target_node_right, y_target_node_right, original_ser=True)
        node, b = su.find_parent(dec_tree, new_node_right)

    else:
        new_node_left, bool_no_red_l = SER(dec_tree.tree_.children_left[node], dec_tree, X_target_node_left, y_target_node_left,
                                           original_ser=False, no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                                           no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify,
                                           leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs, root_source_values=root_source_values,
                                           n_k_min=n_k_min)

        node, b = su.find_parent(dec_tree, new_node_left)

        new_node_right, bool_no_red_r = SER(dec_tree.tree_.children_right[node], dec_tree, X_target_node_right, y_target_node_right, original_ser=False,
                                            no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                                            no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify,
                                            leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs, root_source_values=root_source_values,
                                            n_k_min=n_k_min)

        node, b = su.find_parent(dec_tree, new_node_right)

    if original_ser:
        bool_no_red = False
    else:
        bool_no_red = bool_no_red_l or bool_no_red_r

    leaf_error = su.leaf_error(dec_tree.tree_, node)
    subtree_error = su.subtree_error(dec_tree.tree_, node)
    if DEBUG:
        print(f"LEAF ERROR: {leaf_error}")
        print(f"SUBTREE ERROR: {subtree_error}")
    # if e != 0:
        # raise ValueError("TEST: SUBTREE ERROR IS NON ZERO")

    if leaf_error <= subtree_error:
        if original_ser:
            new_node_leaf = su.cut_into_leaf2(dec_tree, node)
            node = new_node_leaf
        else:
            if no_red_on_cl:
                if not bool_no_red:
                    new_node_leaf = su.cut_into_leaf2(dec_tree, node)
                    node = new_node_leaf
                else:
                    if DEBUG:
                        print('avoid pruning')
            else:
                new_node_leaf = su.cut_into_leaf2(dec_tree, node)
                node = new_node_leaf

    if dec_tree.tree_.feature[node] != -2:
        if original_ser:
            if ind_left.size == 0:
                node = su.cut_from_left_right(dec_tree, node, -1)

            if ind_right.size == 0:
                node = su.cut_from_left_right(dec_tree, node, 1)
        else:
            if no_red_on_cl:
                if ind_left.size == 0 and np.sum(dec_tree.tree_.value[dec_tree.tree_.children_left[node]]) == 0:
                    node = su.cut_from_left_right(dec_tree, node, -1)

                if ind_right.size == 0 and np.sum(dec_tree.tree_.value[dec_tree.tree_.children_right[node]]) == 0:
                    node = su.cut_from_left_right(dec_tree, node, 1)
            else:
                if ind_left.size == 0:
                    node = su.cut_from_left_right(dec_tree, node, -1)

                if ind_right.size == 0:
                    node = su.cut_from_left_right(dec_tree, node, 1)

    return node, bool_no_red


def SER_RF(random_forest, X_target, y_target, original_ser=True, bootstrap_=False,
           no_red_on_cl=False, cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None,
           ext_cond=False, leaf_loss_quantify=False, leaf_loss_threshold=0.9):

    rf_ser = copy.deepcopy(random_forest)

    # TODO consider using dec_tree instead of rf_ser.estimators_[i]
    for i, _dec_tree in enumerate(rf_ser.estimators_):
        root_source_values = None
        coeffs = None
        n_k_min = None
        if leaf_loss_quantify:
            n_k_min = sum(y_target == cl_no_red)
            root_source_values = su.get_node_distribution(rf_ser.estimators_[i], 0).reshape(-1)

            props_s = root_source_values
            props_s = props_s / sum(props_s)
            props_t = np.zeros(props_s.size)
            for k in range(props_s.size):
                props_t[k] = np.sum(y_target == k) / y_target.size

            coeffs = np.divide(props_t, props_s)

            # source_values_tot = rf_ser.estimators_[i].tree_.value[0,0,cl_no_red]

        indices = np.linspace(0, y_target.size - 1, y_target.size).astype(int)
        if DEBUG:
            print(f"Indices: {indices}")
        
        if bootstrap_:
            indices = su.bootstrap(y_target.size)

        SER(0, rf_ser.estimators_[i], X_target[indices], y_target[indices],
            original_ser=original_ser, no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
            no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, ext_cond=ext_cond,
            leaf_loss_quantify=leaf_loss_quantify, leaf_loss_threshold=leaf_loss_threshold,
            coeffs=coeffs, root_source_values=root_source_values, n_k_min=n_k_min)

    return rf_ser
