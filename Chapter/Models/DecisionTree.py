import numpy as np


class DTree(object):

    def __init__(self, alg='ID3', alpha=0.1, prune=True):
        self._category = None
        self._name = None
        self._value = None
        self._majority = None
        self._idx = None
        self._type = alg
        self._alpha = alpha
        self._prune = prune

    def _load(self, X, y, feature=None):
        self._X = X.astype(str)
        self._y = y
        if feature:
            self._feature = feature
        else:
            self._feature = ['f%s'%i for i in range(X.shape[1])]

    def _rm(self):
        if hasattr(self, '_X'):
            delattr(self, '_X')
        if hasattr(self, '_y'):
            delattr(self, '_y')

    def _entropy(self, v):
        L = len(v)
        _, counts = np.unique(v, return_counts=True)
        p = counts / L
        H = - (p*np.log(p)).sum()
        return H

    def _metrics(self, x, y, H_y):
        L = len(y)
        cat, counts = np.unique(x, return_counts=True)
        p = counts / L
        H_Di = 1.0*np.array([self._entropy(y[x == c]) for c in cat])
        H_xy = (p*H_Di).sum()
        info_gain = H_y - H_xy
        if self._type == 'ID3':
            return info_gain
        elif self._type == 'C4.5':
            H_x = self._entropy(x)
            if H_x == 0:
                return np.inf
            else:
                return info_gain / H_x

    def _select(self):
        H_y = self._entropy(self._y)
        max_info = -np.inf
        idx = 0
        for i in range(self._X.shape[1]):
            x = self._X[:, i]
            info_gain = self._metrics(x, self._y, H_y)
            if max_info < info_gain:
                max_info, idx = info_gain, i
        self._idx = idx
        if self._feature != []:
            self._name = self._feature[idx]
            self._category = np.unique(self._X[:, idx])

    def _comp_leaf(self):
        return np.argmax(np.bincount(self._y))

    def _sub_tree(self):
        self._em_entropy = len(self._y) * self._entropy(self._y)
        if self._feature == [] or \
           len(np.unique(self._y)) == 1 or \
           len(np.unique(self._X, axis=0)) == 1:
            self._value = self._comp_leaf()
        else:
            self._majority = self._comp_leaf()
            self._feature.remove(self._name)
            for c in self._category:
                sub_idx = (self._X[:, self._idx] == c)
                sub_X = np.delete(self._X[sub_idx], self._idx, axis=1)
                sub_y = self._y[sub_idx]
                setattr(self, 't%s' % c, DTree(self._type, self._alpha))
                getattr(self, 't%s' % c).fit(sub_X, sub_y)
                if self._prune:
                    setattr(self, 't%s' % c,
                            self._pruning(getattr(self, 't%s' % c)))

    def _pruning(self, tree):
        if tree._value is not None:
            return tree
        H_leaves = 0
        H_root = tree._em_entropy
        alpha = self._alpha
        num_leaves = len(tree._category)
        for c in tree._category:
            if getattr(tree, 't%s' % c)._value is not None:
                return tree
            H_leaves += getattr(tree, 't%s' % c)._em_entropy
        delta = H_root - H_leaves + alpha * (1-num_leaves)
        if delta < 0:
            tree._value = tree._majority
            for c in tree._category:
                delattr(tree, 't%s' % c)
                return tree
        else:
            return tree

    def _predict(self, x):
        x = x.astype(str)
        if self._value is not None:
            return self._value
        else:
            return getattr(self, 't%s' % x[self._idx]).predict(np.delete(x, self._idx))

    def fit(self, X, y):
        self._load(X, y)
        self._select()
        self._sub_tree()
        self._rm()

    def predict(self, X):
        if len(X.shape) == 1: 
            X = X.reshape(1, X.shape[0])
        pred_y = np.array(list(map(self._predict, X)))
        return pred_y.reshape(-1)

    def score(self, X, y):
        return (self.predict(X) == y).sum() / len(y)


class CartTree(object):

    def __init__(self, max_sample=1, max_metrics=0):
        self._category = None
        self._name = None
        self._value = None
        self._majority = None
        self._idx = None
        self._dtype = None
        self._split_value = None
        self._classifier = None
        self._max_sample = max_sample
        self._max_metrics = max_metrics
        self._ltree = None
        self._rtree = None
        self._loss = None

    def _load(self, X, y, feature=None, classifier=None):
        self._X = X
        self._y = y
        if feature:
            self._feature = feature
        else:
            self._feature = ['f%s'%i for i in range(X.shape[1])]
        if classifier is not None:
            self._classifier = classifier
        else:
            if len(np.unique(y)) / len(y) > 0.3:
                self._classifier = False
            else:
                self._classifier = True
                self._adjust = self._y.min()
                self._y = (self._y-self._y.min()).astype(int)

    def _rm(self):
        if hasattr(self, '_X'):
            delattr(self, '_X')
        if hasattr(self, '_y'):
            delattr(self, '_y')

    def _gini(self, x):
        _, c = np.unique(x, return_counts=True)
        gini = 1 - ((c / x.shape[0]) ** 2).sum()
        return gini

    def _fea_select(self, X, y):
        f = -1
        metrics = np.inf
        for x in X.T:
            f += 1
            is_discrete = (len(np.unique(x)) < 0.3*len(x))
            space = np.unique(x) if is_discrete else x
            for x_i in space:
                if is_discrete:
                    D1 = (x == x_i)
                    D2 = ~ D1
                    if self._classifier:
                        metrics_i = D1.sum() * self._gini(y[D1]) \
                                    + D2.sum() * self._gini(y[D2])
                    else:
                        metrics_i = y[D1].mean() + y[D2].mean()
                else:
                    D1 = (x < x_i)
                    D2 = ~ D1
                    if self._classifier:
                        metrics_i = D1.sum() * self._gini(y[D1]) \
                                    + D2.sum() * self._gini(y[D2])
                    else:
                        metrics_i = y[D1].mean() + y[D2].mean()

            if metrics_i < metrics:
                idx = f
                v = x_i
                metrics = metrics_i

        return idx, v, is_discrete, metrics

    def _gen_tree(self):
        if len(self._y) < self._max_sample \
           or len(np.unique(self._y)) == 1 \
           or len(np.unique(self._X.astype(str), axis=0)) == 1:
            if self._classifier:
                if len(self._y) == 0: 
                    self._loss = 0
                    return
                self._value = np.argmax(np.bincount(self._y))
                D1 = (self._y == self._value)
                D2 = ~D1
                self._loss = self._gini(self._y[D1]) * D1.sum() \
                            + self._gini(self._y[D2]) * D2.sum()
            else:
                self._value = self._y.mean()
                self._loss = ((self._y - self._value)**2).sum()
            return

        self._idx, self._split_value, self._dtype, metrics = self._fea_select(self._X, self._y)
        if metrics < self._max_metrics:
            if self._classifier:
                self._value = np.argmax(np.bincount(self._y))
                D1 = (self._y == self._value)
                D2 = ~D1
                self._loss = self._gini(self._y[D1]) * D1.sum() \
                            + self._gini(self._y[D2]) * D2.sum()
            else:
                self._value = self._y.mean()
                self._loss = ((self._y - self._value)**2).sum()
            return

        if self._classifier:
            self._majority = np.argmax(np.bincount(self._y))
            D1 = (self._y == self._majority)
            D2 = ~D1
            self._loss = self._gini(self._y[D1]) * D1.sum() \
                         + self._gini(self._y[D2]) * D2.sum()
        else:
            self._majority = self._y.mean()
            self._loss = ((self._y - self._majority)**2).sum()
        
        if self._dtype:
            t0 = (self._X[:, self._idx] == self._split_value)
            t1 = ~t0
        else:
            t0 = (self._X[:, self._idx] < self._split_value)
            if t0.sum() == 0: 
                t0 = (self._X[:, self._idx] <= self._split_value)
            t1 = ~t0

        self._ltree = CartTree(self._max_sample, self._max_metrics)
        self._ltree._load(self._X[t0], self._y[t0], classifier=self._classifier)
        self._ltree._gen_tree()

        self._rtree = CartTree(self._max_sample, self._max_metrics)
        self._rtree._load(self._X[t1], self._y[t1], classifier=self._classifier)
        self._rtree._gen_tree()

        self._rm()

    def _predict(self, x):
        if self._value is not None:
            return self._value
        else:
            if self._dtype:
                if x[self._idx] == self._split_value:
                    return self._ltree._predict(x)
                else:
                    return self._rtree._predict(x)
            else:
                if x[self._idx] < self._split_value:
                    return self._ltree._predict(x)
                else:
                    return self._rtree._predict(x)

    def fit(self, X, y):
        self._load(X, y)
        self._gen_tree()

    def predict(self, X):
        if len(X.shape) == 1: 
            X = X.reshape(1, X.shape[0])
        pred_y = np.array(list(map(self._predict, X)))
        return (pred_y.reshape(-1)+self._adjust).astype(int)

    def score(self, X, y):
        return (self.predict(X) == y).sum() / len(y)


class DecisionTree(object):

    def __init__(self, alg='ID3', alpha=0, max_sample=1, prune=True):
        if alg in ['ID3', 'C4.5']:
            self._tree = DTree(alg, alpha, prune=prune)
        elif alg == 'CART':
            self._tree = CartTree(max_sample, alpha)

    def fit(self, X, y):
        self._tree.fit(X, y)

    def predict(self, X):
        return self._tree.predict(X)

    def score(self, X, y):
        return self._tree.score(X, y)