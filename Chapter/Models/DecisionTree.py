import numpy as np

class TreeID3(object):
    def __init__(self):
        self._category = None
        self._name = None
        self._value = None
        self._idx = None

    def _load(self, X, y, feature=None):
        self._X = X.astype(str)
        self._y = y
        if feature:
            self._feature = feature
        else:
            self._feature = ['f%s'%i for i in range(X.shape[1])]

    def _rm(self):
        if hasattr(self, '_X'): delattr(self, '_X')
        if hasattr(self, '_y'): delattr(self, '_y')

    def _entropy(self, v):
        L = len(v)
        _, counts = np.unique(v, return_counts=True)
        p = counts / L
        H = - (p*np.log(p)).sum()
        return H

    def _info_gain(self, x, y, H_y):
        L = len(y)
        cat, counts = np.unique(x, return_counts=True)
        p = counts / L
        H_Di = 1.0*np.array([self._entropy(y[x==c]) for c in cat])
        H_xy = (p*H_Di).sum()
        info_gain = H_y - H_xy
        return info_gain, cat

    def _select(self):
        H_y = self._entropy(self._y)
        max_info = -np.inf
        idx = 0
        for i in range(self._X.shape[1]):
            x = self._X[:, i]
            info_gain, cat = self._info_gain(x, self._y, H_y)
            if max_info < info_gain:
                max_info, idx = info_gain, i
        self._idx = idx
        self._name = self._feature[idx]
        self._feature.remove(self._name)
        self._category = np.unique(self._X[:, idx])

    def _comp_leaf(self):
        return np.argmax(np.bincount(self._y))

    def _sub_tree(self):
        if self._feature == []:
            self._value = self._comp_leaf()
        elif len(np.unique(self._y)) == 1:
            self._value = self._comp_leaf()
        else:
            for c in self._category:
                sub_idx = (self._X[:, self._idx] == c)
                sub_X = np.delete(self._X[sub_idx], self._idx, axis=1)
                sub_y = self._y[sub_idx]
                setattr(self, 'f%s'%c, TreeID3())
                getattr(self, 'f%s'%c).fit(sub_X, sub_y)

    def predict(self, x):
        x = x.astype(str)
        if not self._value is None: 
            return self._value
        else:
            return getattr(self, 'f%s'%x[self._idx]).predict(np.delete(x, self._idx))

    def fit(self, X, y):
        self._load(X, y)
        self._select()
        self._sub_tree()
        self._rm()
