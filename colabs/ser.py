import copy
import typing as ty
import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm


def depth_vtree(tree, node):
  p, _, _ = extract_rule_vtree(tree, node)
  return len(p)


def fuse_trees(tree_1, leaf_f, tree_2):
  """adding tree tree_2 to leaf_f of tree tree_1"""
  
  dic = tree_1.__getstate__().copy()
  dic2 = tree_2.__getstate__().copy()

  size_init = tree_1.node_count
  if depth_vtree(tree_1, leaf_f) + dic2['max_depth'] > dic['max_depth']:
    dic['max_depth'] = depth_vtree(tree_1, leaf_f) + tree_2.max_depth
  
  dic['capacity'] = tree_1.capacity + tree_2.capacity - 1
  dic['node_count'] = tree_1.node_count + tree_2.node_count - 1
  
  dic['nodes'][leaf_f] = dic2['nodes'][0]
  
  if (dic2['nodes']['left_child'][0] != - 1):
    dic['nodes']['left_child'][leaf_f] = dic2['nodes']['left_child'][0] + size_init - 1
  else:
    dic['nodes']['left_child'][leaf_f] = -1
  
  if (dic2['nodes']['right_child'][0] != - 1):
    dic['nodes']['right_child'][leaf_f] = dic2['nodes']['right_child'][0] + size_init - 1
  else:
    dic['nodes']['right_child'][leaf_f] = -1
  
  # Attention vector impurity not updated
  dic['nodes'] = np.concatenate((dic['nodes'], dic2['nodes'][1:]))
  
  left_child = dic['nodes']['left_child'][size_init:]
  right_child = dic['nodes']['right_child'][size_init:]

  dic['nodes']['left_child'][size_init:] = (left_child != -1) * (left_child + size_init) - 1
  dic['nodes']['right_child'][size_init:] = (right_child != -1) * (right_child + size_init) - 1
  
  values = np.concatenate((
    dic['values'], 
    np.zeros((
      dic2['values'].shape[0] - 1, 
      dic['values'].shape[1], 
      dic['values'].shape[2]))), axis=0)
  
  dic['values'] = values
  
  # Attention :: (potentially important)
  (Tree, (n_f, n_c, n_o), b) = tree_1.__reduce__()
  tree_1 = Tree(n_f, n_c, n_o)
  tree_1.__setstate__(dic)

  return tree_1


def fuse_dec_trees(dec_tree, leaf_f, new_dec_tree):
  """adding tree new_dec_tree to leaf_f of tree dec_tree"""
  size_init = dec_tree.tree_.node_count
  dec_tree.tree_ = fuse_trees(dec_tree.tree_, leaf_f, new_dec_tree.tree_)
  
  try:
    dec_tree.tree_.value[size_init:, :, new_dec_tree.classes_.astype(int)] = new_dec_tree.tree_.value[1:, :, :]
  except IndexError as e:
    print("IndexError : size init : ", size_init, "\ndTree2.classes_ : ", new_dec_tree.classes_)
    print(e)
  
  dec_tree.max_depth = dec_tree.tree_.max_depth
  return dec_tree

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

def find_parent(dec_tree, i_node):
    return find_parent_vtree(dec_tree.tree_, i_node)

def add_to_parents(dec_tree, node, original_tree_vals):
  p, b = find_parent(dec_tree, node)

  if b != 0:
    dec_tree.tree_.value[p] = dec_tree.tree_.value[p] + original_tree_vals
    add_to_parents(dec_tree, p, original_tree_vals)

def get_leaf_error(tree, node):
  tree_vals = tree.value[node]
  if np.sum(tree_vals) == 0:
      return 0
  else:
      return 1 - np.max(tree_vals) / np.sum(tree_vals)
      
def get_node_distribution(dec_tree, node_index):
  tree = dec_tree.tree_
  Q = tree.value[node_index]
  return np.asarray(Q)

def get_subtree_error(tree, node):
  if node == -1:
    return 0
  if tree.feature[node] == -2:
    return get_leaf_error(tree, node)
  else:
    # Not a leaf
    nr = np.sum(tree.value[tree.children_right[node]])
    nl = np.sum(tree.value[tree.children_left[node]])

    if nr + nl == 0:
      return 0
    else:
      er = get_subtree_error(tree, tree.children_right[node])
      el = get_subtree_error(tree, tree.children_left[node])

      return (el * nl + er * nr) / (nl + nr)


def sub_nodes(tree, node):
  if (node == -1):
    return list()
  if (tree.feature[node] == -2):
    return [node]
  else:
    return [node] + sub_nodes(tree, tree.children_left[node]) + sub_nodes(tree, tree.children_right[node])


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
      try:
        node, b = find_parent_vtree(tree, node)
      except:
        print(node, b)

    feats.pop(0)
    ths.pop(0)
    bools.pop(0)
    nodes.pop(0)

  return np.array(feats), np.array(ths), np.array(bools)


def extract_rule(dec_tree, i_node):
  return extract_rule_vtree(dec_tree.tree_, i_node)


def depth(dec_tree, node):
  p, _, _ = extract_rule(dec_tree, node)
  return len(p)

def depth_array(dec_tree, indices):
  depths = np.zeros(np.array(indices).size)
  for idx, _ in enumerate(indices):
    depths[idx] = depth(dec_tree, idx)
  return depths


def cut_into_leaf2(dec_tree, node):
  dic = dec_tree.tree_.__getstate__().copy()

  node_to_rem = list()
  size_init = dec_tree.tree_.node_count

  node_to_rem = node_to_rem + sub_nodes(dec_tree.tree_, node)[1:]
  node_to_rem = list(set(node_to_rem))

  indices = list(
    set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))
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


def cut_from_left_right(dec_tree, node, bool_left_right):
  dic = dec_tree.tree_.__getstate__().copy()

  node_to_rem = list()
  size_init = dec_tree.tree_.node_count

  p, b = find_parent(dec_tree, node)

  if bool_left_right == 1:
    repl_node = dec_tree.tree_.children_left[node]
    node_to_rem = [node, dec_tree.tree_.children_right[node]]
  elif bool_left_right == -1:
    repl_node = dec_tree.tree_.children_right[node]
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

  depths = depth_array(
    dec_tree, np.linspace(
      0, dec_tree.tree_.node_count - 1, dec_tree.tree_.node_count).astype(int))
  dec_tree.tree_.max_depth = np.max(depths)

  return indices.index(repl_node)


def pad_to_dense(M, maxlen=100):
  """thx to https://stackoverflow.com/questions/37676539:
  Appends the minimal required amount of zeroes at the end of each 
  array in the jagged array `M`, such that `M` looses its jaggedness.
  """
  Z = np.zeros((len(M), maxlen))
  for enu, row in enumerate(M):
    Z[enu, :len(row)] += row 
  return Z


class Options:
  def __init__(self, **kwargs) -> None:
    # thx to https://stackoverflow.com/questions/5899185
    prop_defaults = {
      'cl_no_red': None,
      'leaf_loss_quantify': False,
      'no_red_on_cl': False,
      'no_ext_on_cl': False,
      'cl_no_ext': None,
      'root_source_values': None,
      'leaf_loss_threshold': None,
      'n_k_min': None,
      'coeffs': None,
      'no_red_on_cl': False,
      'original_ser': False,
      'ext_cond': None,}
    
    self.__dict__.update(prop_defaults)
    self.__dict__.update(kwargs)


def ser(node, dec_tree, X_tgt, y_tgt, extra: Options = Options()):
  # Deep copy of value
  old_tree_vals =  dec_tree.tree_.value[node].copy()
  maj_cls = np.argmax(dec_tree.tree_.value[node, :])

  old_size_cl_no_red = 0
  if extra.cl_no_red is not None:
    old_size_cl_no_red = np.sum(dec_tree.tree_.value[node][:, extra.cl_no_red])

  if (extra.leaf_loss_quantify and
      (extra.no_red_on_cl or extra.no_ext_on_cl
        and dec_tree.tree_.feature[node] == -2)):
    cl = extra.cl_no_ext[0]

    if extra.no_red_on_cl:
        cl = extra.cl_no_red[0]

    ps_rf = dec_tree.tree_.value[node, 0, :] / sum(dec_tree.tree_.value[node, 0, :])
    p1_in_l = dec_tree.tree_.value[node, 0, cl] / extra.root_source_values[cl]

    # output of new function
    cond1 = np.power(1 - p1_in_l, extra.n_k_min) > extra.leaf_loss_threshold
    cond2 = np.argmax(np.multiply(extra.coeffs, ps_rf)) == cl

  ### value updates ###
  vals_to_update = np.zeros((dec_tree.n_outputs_, dec_tree.n_classes_))
  for idx in range(dec_tree.n_classes_):
    vals_to_update[:, idx] = list(y_tgt).count(idx)

  dec_tree.tree_.value[node] = vals_to_update
  dec_tree.tree_.n_node_samples[node] = np.sum(vals_to_update)
  dec_tree.tree_.weighted_n_node_samples[node] = np.sum(vals_to_update)

  ### Class resolution ###
  if dec_tree.tree_.feature[node] == -2:
    if extra.original_ser:
      if y_tgt.size > 0 and len(set(list(y_tgt))) > 1:
        # the class changes automatically according to
        # target by the values ​​updates
        dec_tree_to_add = DecisionTreeClassifier()
        try:
            dec_tree_to_add.min_impurity_decrease = 0
        except:
            dec_tree_to_add.min_impurity_split = 0
        dec_tree_to_add.fit(X_tgt, y_tgt)
        fuse_dec_trees(dec_tree, node, dec_tree_to_add)
      return node, False
    else:
      bool_no_red = False
      cond_extension = False

      if y_tgt.size > 0:
        # Extension
        if not extra.no_ext_on_cl:
          dec_tree_to_add = DecisionTreeClassifier()
          # make a complete tree
          try:
              dec_tree_to_add.min_impurity_decrease = 0
          except:
              dec_tree_to_add.min_impurity_split = 0
          dec_tree_to_add.fit(X_tgt, y_tgt)
          fuse_dec_trees(dec_tree, node, dec_tree_to_add)
        else:
          cond_maj = (maj_cls not in extra.cl_no_ext)
          cond_sub_target = extra.ext_cond and (maj_cls in y_tgt) and (maj_cls in extra.cl_no_ext)
          cond_leaf_loss = extra.leaf_loss_quantify and not (cond1 and cond2)

          cond_extension = cond_maj or cond_sub_target or cond_leaf_loss
          if cond_extension:
            dec_tree_to_add = DecisionTreeClassifier()
            # to make a complete tree
            try:
                dec_tree_to_add.min_impurity_decrease = 0
            except:
                dec_tree_to_add.min_impurity_split = 0
            dec_tree_to_add.fit(X_tgt, y_tgt)
            fuse_dec_trees(dec_tree, node, dec_tree_to_add)
          else:
            # Complicated not to induce any inconsistency in the
            # values ​​by leaving the sheets intact in this way.
            # That said, it has no impact on the decision tree
            # that we want to obtain (it has one on the probability tree
            dec_tree.tree_.value[node] = old_tree_vals
            dec_tree.tree_.n_node_samples[node] = np.sum(old_tree_vals)
            dec_tree.tree_.weighted_n_node_samples[node] = np.sum(old_tree_vals)
            add_to_parents(dec_tree, node, old_tree_vals)
            if extra.no_red_on_cl:
              bool_no_red = True

      # no reduction protection with values
      if (extra.no_red_on_cl and y_tgt.size == 0
          and old_size_cl_no_red > 0 and maj_cls in extra.cl_no_red):
        if extra.leaf_loss_quantify:
          if cond1 and cond2:
            dec_tree.tree_.value[node] = old_tree_vals
            dec_tree.tree_.n_node_samples[node] = np.sum(old_tree_vals)
            dec_tree.tree_.weighted_n_node_samples[node] = np.sum(old_tree_vals)
            add_to_parents(dec_tree, node, old_tree_vals)
            bool_no_red = True
          else:
            dec_tree.tree_.value[node] = old_tree_vals
            dec_tree.tree_.n_node_samples[node] = np.sum(old_tree_vals)
            dec_tree.tree_.weighted_n_node_samples[node] = np.sum(old_tree_vals)
            add_to_parents(dec_tree, node, old_tree_vals)
            bool_no_red = True

      return node, bool_no_red

  ### Left / right target computation ###
  bool_test = X_tgt[:, dec_tree.tree_.feature[node]] <= dec_tree.tree_.threshold[node]
  not_bool_test = X_tgt[:, dec_tree.tree_.feature[node]] > dec_tree.tree_.threshold[node]

  ind_left = np.where(bool_test)[0]
  ind_right = np.where(not_bool_test)[0]

  X_tgt_left = X_tgt[ind_left]
  y_tgt_left = y_tgt[ind_left]

  X_tgt_right = X_tgt[ind_right]
  y_tgt_right = y_tgt[ind_right]

  if extra.original_ser:
    extra.original_ser = True
    new_node_left, bool_no_red_l = ser(
      dec_tree.tree_.children_left[node], dec_tree,
      X_tgt_left, y_tgt_left, extra)

    node, b = find_parent(dec_tree, new_node_left)

    new_node_right, bool_no_red_r = ser(
      dec_tree.tree_.children_right[node], dec_tree,
      X_tgt_right, y_tgt_right, extra)

    node, b = find_parent(dec_tree, new_node_right)
  else:
    extra.original_ser = False
    new_node_left, bool_no_red_l = ser(
      dec_tree.tree_.children_left[node], dec_tree, X_tgt_left, y_tgt_left, extra=extra)
    
    node, b = find_parent(dec_tree, new_node_left)
    
    new_node_right, bool_no_red_r = ser(
      dec_tree.tree_.children_right[node], dec_tree, X_tgt_right, y_tgt_right, extra=extra)
    
    node, b = find_parent(dec_tree, new_node_right)

  bool_no_red = False
  if not extra.original_ser:
      bool_no_red = bool_no_red_l or bool_no_red_r

  leaf_error = get_leaf_error(dec_tree.tree_, node)
  subtree_error = get_subtree_error(dec_tree.tree_, node)

  if leaf_error <= subtree_error:
    if extra.original_ser:
      new_node_leaf = cut_into_leaf2(dec_tree, node)
      node = new_node_leaf
    else:
      if extra.no_red_on_cl:
        # Prune tree
        if not bool_no_red:
          new_node_leaf = cut_into_leaf2(dec_tree, node)
          node = new_node_leaf
        else:
          print("avoid pruning tree")
      else:
        new_node_leaf = cut_into_leaf2(dec_tree, node)
        node = new_node_leaf
  
  if dec_tree.tree_.feature[node] != -2:
    if extra.original_ser:
      if ind_left.size == 0:
        node = cut_from_left_right(dec_tree, node, -1)
      if ind_right.size == 0:
        node = cut_from_left_right(dec_tree, node, 1)
    else:
      if extra.no_red_on_cl:
        if (ind_left.size == 0 
            and np.sum(dec_tree.tree_.value[dec_tree.tree_.children_left[node]]) == 0):
          node = cut_from_left_right(dec_tree, node, -1)
        if (ind_right.size == 0 
            and np.sum(dec_tree.tree_.value[dec_tree.tree_.children_right[node]]) == 0):
          node = cut_from_left_right(dec_tree, node, 1)
      else:
        if ind_left.size == 0:
          node = cut_from_left_right(dec_tree, node, -1)
        if ind_right.size == 0:
          node = cut_from_left_right(dec_tree, node, 1)

  return node, bool_no_red


class TreesTransferLearning(BaseEstimator, ClassifierMixin):
  def __init__(
    self, 
    module: ty.Union[RandomForestClassifier, DecisionTreeClassifier],
    **kwargs
    ) -> None:
    
    if module is None:
      raise ValueError("module is None")
    
    self.is_rf_clf = isinstance(module, RandomForestClassifier)
    self.module = copy.deepcopy(module)
    self.options = Options(**kwargs)
  
  def fit(self, X, y, *fit_params):
    if X is None or y is None:
      raise ValueError("missing X,y data")
    
    if fit_params:
      # "re" fit the original classifier
      param_dict = fit_params[:]  # shallow copy
      if len(param_dict) != 2:
        raise ValueError("requires only two params")
    
      # fit non-transfer-learned classifier
      self.module.fit(*fit_params)
    
    # Applies SER algorithm to either RandomForestClassifier
    # or DecisionTreeClassifier
    if self.is_rf_clf:
      # work on random forests
      self._finetune_rf_model(X, y)
    else:
      # work on decision trees
      self._finetune_dt_model(X, y)

    return self
  
  def score(self, X, y):
    return self.module.score(X, y)
  
  def _finetune_dt_model(self, X, y):
    ser(0, self.module, X, y, extra = self.options)

    return self 
  
  def _finetune_rf_model(self, X, y):
    
    tk = tqdm(self.module.estimators_, desc="tune_rf",) 
    t_i = 0
    for idx, _ in enumerate(tk):
      root_source_values = None
      coeffs = None
      n_k_min = None
      if self.options.leaf_loss_quantify:
        n_k_min = sum(y == self.options.cl_no_red)
        root_source_values = get_node_distribution(
          self.module.estimators_[idx], 0).reshape(-1)
        
        props_s = root_source_values
        props_s = props_s / sum(props_s)
        props_t = np.zeros(props_s.size)
        
        for k in range(props_s.size):
          props_t[k] = np.sum(y == k) / y.size
        
        coeffs = np.divide(props_t, props_s)
        
      indices = np.linspace(0, y.size - 1, y.size).astype(int)
      if self.options.bootstrap_:
        # bootstrap indices
        indices = np.random.choice(
          np.linspace(0, y.size - 1, y.size).astype(int), y.size, replace=True)

      self.options.coeffs = coeffs
      self.options.n_k_min = n_k_min
      self.root_source_values = root_source_values
      
      ser(0, self.module.estimators_[idx], X[indices], y[indices], extra = self.options)
      tk.set_postfix({"trial": f"{t_i}",})
      t_i += 1
    
    return self
  
  def predict(self, X):
    y_pred = self.module.predict(X)
    return y_pred
    
  def predict_proba(self, X):
    return self.module.predict_proba(X)
  

class TestTreesTransferLearning(unittest.TestCase):
  @staticmethod
  def build_synthetic_data():
    # Generate training source data
    np.random.seed(0)
    
    n_samples = 200
    n_samples_perclass = n_samples // 2

    mean_1 = (1, 1)
    var_1 = np.diag([1, 1])
    mean_2 = (3, 3)
    var_2 = np.diag([2, 2])
    
    X_src = np.r_[
      np.random.multivariate_normal(mean_1, var_1, size=n_samples_perclass), 
      np.random.multivariate_normal(mean_2, var_2, size=n_samples_perclass)]

    y_src = np.zeros(n_samples)
    y_src[n_samples_perclass:] = 1

    # Generate training target data
    nt = 50
    
    # imbalanced
    nt_0 = nt // 10
    mean_1 = (6, 3)
    var_1 = np.diag([4, 1])
    mean_2 = (5, 5)
    var_2 = np.diag([1, 3])
    
    X_tgt = np.r_[
      np.random.multivariate_normal(mean_1, var_1, size=nt_0), 
      np.random.multivariate_normal(mean_2, var_2, size=nt - nt_0)]
    y_tgt = np.zeros(nt)
    y_tgt[nt_0:] = 1
    
    # Generate testing target data
    nt_test = 1000
    nt_test_perclass = nt_test // 2
    
    X_tgt_test = np.r_[
      np.random.multivariate_normal(mean_1, var_1, size=nt_test_perclass), 
      np.random.multivariate_normal(mean_2, var_2, size=nt_test_perclass)]
    y_tgt_test = np.zeros(nt_test)
    y_tgt_test[nt_test_perclass:] = 1
    
    return X_src, y_src, X_tgt, y_tgt, X_tgt_test, y_tgt_test
  
  @staticmethod
  def get_digits_data():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    digits = load_digits()
    
    np.random.seed(0)
    
    X = digits.data[:200]
    y = (digits.target[:200] == 6).astype(int)
    
    X_tgt = digits.data[200:]
    y_tgt = (digits.target[200:] == 9).astype(int)
    
    # separating 5% & 95% of target data, stratified, random
    X_tgt_095, X_tgt_005, y_tgt_095, y_tgt_005 = train_test_split(
      X_tgt, y_tgt, test_size=0.05, stratify=y_tgt)
    
    return X, X_tgt_005, X_tgt_095, y, y_tgt_005, y_tgt_095


  def test_dec_tree_classifier(self):
    X_src, y_src, _, _, X_tgt_test, y_tgt_test = TestTreesTransferLearning.build_synthetic_data()
    dec_tree_src = DecisionTreeClassifier(max_depth=None)
    dec_tree_src.fit(X_src, y_src)
    score_src_src = dec_tree_src.score(X_src, y_src)
    score_src_tgt = dec_tree_src.score(X_tgt_test, y_tgt_test)
    
    print('Training score Source model: {:.3f}'.format(score_src_src))
    print('Testing score Source model: {:.3f}'.format(score_src_tgt))
    
    self.assertEqual(1.000, score_src_src)
    self.assertEqual(0.516, score_src_tgt)
  
  def test_options(self):
    opts = Options()
    self.assertIsNotNone(opts)
    self.assertIsNone(opts.coeffs)
    self.assertFalse(opts.original_ser)
    self.assertFalse(opts.no_red_on_cl)
    
  def test_tl_dec_tree_classifier(self):
    X_src, y_src, X_tgt, y_tgt, X_tgt_test, y_tgt_test = TestTreesTransferLearning.build_synthetic_data()
    
    # transfer learning (with reduction) version of dec trees (ser original)
    dec_tree_tgt = TreesTransferLearning(DecisionTreeClassifier(max_depth=None))
    self.assertIsNotNone(dec_tree_tgt)
    dec_tree_tgt.fit(X_tgt, y_tgt, X_src, y_src)
    score_tgt_tgt = dec_tree_tgt.score(X_tgt_test, y_tgt_test)

    print('Testing score transferred model ({}) : {:.3f}'.format("ser", score_tgt_tgt))
    
    self.assertEqual(0.647, score_tgt_tgt)
    dec_tree_tgt = None
    
    
  def test_random_forest_with_reduction(self):
    MAX, N_EST = 5, 3
    
    X_src, X_tgt_005, X_tgt_095, y_src, y_tgt_005, y_tgt_095 = TestTreesTransferLearning.get_digits_data()
    rand_forest_tgt = TreesTransferLearning(RandomForestClassifier(n_estimators=N_EST, max_depth=MAX), bootstrap_=False)
    
    self.assertIsNotNone(rand_forest_tgt)
    
    rand_forest_tgt.fit(X_tgt_005, y_tgt_005, X_src, y_src)
    score_tgt_tgt = rand_forest_tgt.score(X_tgt_095, y_tgt_095)
    
    print("y_pred", rand_forest_tgt.predict(X_tgt_095))

    print('Testing score transferred model (with reduction) ({}) : {:.3f}'.format("ser", score_tgt_tgt))


  def test_transfer_learning(self):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from os.path import dirname, abspath, join
    
    signal_d = dirname(dirname(abspath(__file__)))
    dialog_data = join(signal_d, 'data/dialog_data.csv')

    dialog_dataset_df = pd.read_csv(dialog_data)
    dialog_dataset_df['persuasion_id'] = dialog_dataset_df['persuasion'].factorize()[0]
    
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    X = tfidf.fit_transform(dialog_dataset_df.text).toarray()
    y = dialog_dataset_df.persuasion_id
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size = 0.4, random_state = 42)
    
    orig_model = RandomForestClassifier(n_estimators=20, max_depth=None)
    orig_model.fit(X_train, y_train)
    
    dev_dialog_data = join(signal_d, 'data/dev_dialog_data.csv')
    dev_dialog_data_df = pd.read_csv(dev_dialog_data)
    
    X_tgt = tfidf.fit_transform(dev_dialog_data_df.text).toarray()
    y_tgt = dev_dialog_data_df.persuasion_id.to_numpy()
    
    X_tgt = pad_to_dense(X_tgt, X.shape[1])
    print(X_tgt.shape)
    
    X_tgt_090, X_tgt_010, y_tgt_090, y_tgt_010 = train_test_split(X_tgt, y_tgt, test_size=0.10, stratify=y_tgt)
    
    ttl_model = TreesTransferLearning(orig_model, bootstrap_=True, original_ser=True)
    ttl_model.fit(X_tgt_090, y_tgt_090)

    print("")

  
if __name__ == '__main__':
    unittest.main()
