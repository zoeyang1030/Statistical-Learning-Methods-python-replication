import numpy as np


class DTree(object):

    def __init__(self, alg='ID3', alpha=0.1):
        self._category = None
        self._name = None
        self._value = None
        self._majority = None
        self._idx = None
        self._type = alg
        self._alpha = alpha

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
                setattr(self, 't%s' % c, DTree())
                getattr(self, 't%s' % c).fit(sub_X, sub_y)
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


class DecisionTree(object):

    def __init__(self, alg='ID3', alpha=0.1):
        if alg in ['ID3', 'C4.5']:
            self._tree = DTree(alg, alpha)

    def fit(self, X, y):
        self._tree.fit(X, y)

    def predict(self, X):
        return self._tree.predict(X)

    def score(self, X, y):
        return self._tree.score(X, y)