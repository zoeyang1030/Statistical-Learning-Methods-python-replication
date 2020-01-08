import numpy as np

class KDTree(object):

    __slots__ = ('_node', '_label', '_left', '_right', '_index')

    def __init__(self, node, label, c=0, split=1):
        self._node = node
        self._label = label
        self._left = None
        self._right = None
        self._index = c
        if split: self._separate(c)

    def __getattr__(self, attr):
        if '_%s'%attr in self.__slots__:
            return getattr(self, '_%s'%attr)
        else:
            raise AttributeError("'KDTree' object has no attribute '%s'"%attr)

    def _sort(self, node, label, c):
        sort_index = np.argsort(node[:, c])
        return node[sort_index], label[sort_index]

    def _separate(self, c):
        node = self._node
        label = self._label
        if node.shape[0] == 1: 
            self._node = node[0, :]
            self._label = label[0]
        elif node.shape[0] == 2: 
            node, label = self._sort(node, label, c)
            self._node = node[1, :]
            self._label = label[1]
            c = (c + 1) % node.shape[1]
            self._left = KDTree(node[0, :], label[0], c, 0)
        elif node.shape[0] == 3: 
            node, label = self._sort(node, label, c)
            self._node = node[1, :]
            self._label = label[1]
            c = (c + 1) % node.shape[1]
            self._left = KDTree(node[0, :], label[0], c, 0)
            self._right = KDTree(node[2, :], label[2], c, 0)
        else:
            mid = node.shape[0] // 2 + node.shape[0] % 2 - 1
            node, label = self._sort(node, label, c)
            c = (c + 1) % node.shape[1]           
            self._node = node[mid, :]
            self._label = label[mid]
            self._left = KDTree(node[:mid, :], label[:mid], c)
            self._right = KDTree(node[mid+1:, :], label[mid+1:], c)


class KNearestNeighbor(object):

    def __init__(self, n_neighbors, d=2):
        self._param = {'n_neighbors': n_neighbors, 'ord': d}
        self._tree = None

    def _search_kd_tree(self, tree, point, neighbors):
        tree_list = []
        while tree:  
            tree_list.append(tree)  
            neighbors = self._compare_dist(tree.node, tree.label, point, neighbors)
            tree = tree.right if point[tree.index] >= tree.node[tree.index] else tree.left
        tree_list.pop()
        while tree_list:
            tree = tree_list.pop()
            if abs(point[tree.index] - tree.node[tree.index]) < neighbors[2].max():
                tree = tree.right if point[tree.index] < tree.node[tree.index] else tree.left
                if tree:
                    neighbors = self._search_kd_tree(tree, point, neighbors)
        return neighbors
        
    def _search(self, point):
        if self._tree: tree = self._tree
        else: return

        l = point.shape[0]
        neighbors = [
            np.array([np.array([np.inf]*l)]*self._param['n_neighbors']),
            np.array([0]*self._param['n_neighbors']),
            np.array([np.inf]*self._param['n_neighbors'])
        ]

        neighbors = self._search_kd_tree(tree, point, neighbors)
        
        return neighbors

    def _compare_dist(self, node, label, point, neighbors):
        dist = np.linalg.norm(point-node, ord=self._param['ord'])
        max_d = neighbors[2].max()
        if dist < max_d:
            i = np.argwhere(neighbors[2]==max_d)[0]
            neighbors[0][i] = node
            neighbors[1][i] = label
            neighbors[2][i] = dist
        return neighbors
    
    def _search_label(self, point):
        return (self._search(point)[1].sum()>=0)*1

    def fit(self, X, y):
        self._tree = KDTree(X, y)

    def predict(self, X):
        if len(X.shape) == 1: X = X.reshape(1, X.shape[0])
        return np.apply_along_axis(self._search_label, 1, X)