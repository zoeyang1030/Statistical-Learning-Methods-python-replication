import numpy as np
from scipy import stats


class Bernoulli(object):

    def __init__(self, lamda=1):
        self.lamda = lamda

    def fit(self, X, y):
        if len(X.shape) == 1: X = X.reshape(1, X.shape[0])
        c = list(set(y))
        a = [list(set(X[:, i])) for i in range(X.shape[1])]
        P = {}
        N = len(y)

        P_y = {}
        P_xy = {}
        for c_k in c:
            P_y[c_k] = ((y==c_k).sum() + self.lamda) / (N + self.lamda * len(c))
            P_xy[c_k] = {}
            for j in range(X.shape[1]):
                P_xy[c_k][j] = {}
                for l, a_jl in enumerate(a[j]):
                    P_xy[c_k][j][a_jl] = ((X[y==c_k, j]==a[j][l]).sum() + self.lamda) \
                                        / (P_y[c_k]*N + self.lamda * len(a[j]))
        
        self.P_y = P_y
        self.P_xy = P_xy

    def _predict(self, x):
        x = x.reshape(1, x.shape[0])
        P_y = self.P_y
        P_xy = self.P_xy
        pred_P = np.array([])
        c = list(P_y.keys())
        for c_k in c:
            [P_xy[c_k][j][x[0][j]] for j in range(x.shape[1])]
            p = np.prod([P_y[c_k]]+[P_xy[c_k][j][x[0][j]] for j in range(x.shape[1])])
            pred_P = np.append(pred_P, p)
        return c[np.where(pred_P==pred_P.max())[0][0]]

    def predict(self, X):
        if len(X.shape) == 1: X = X.reshape(1, X.shape[0])
        return np.array(list(map(self._predict, X)))

    def score(X, y):
        return (self.predict(X)==y).sum() / len(y)


class Gaussian(object):

    def __init__(self, prior=None):
        self.prior = prior
        
    def fit(self, X, y):
        if len(X.shape) == 1: X = X.reshape(1, X.shape[0])
        c = list(set(y))
        P_y = {}
        G_param = {}
        for c_k in c:
            P_y[c_k] = (y==c_k).sum() / len(y)
            G_param[c_k] = {}
            for i in range(X.shape[1]):
                G_param[c_k][i] = [X[y==c_k, i].mean(), X[y==c_k, i].std()]
                if G_param[c_k][i][1] == 0:
                    raise Exception('Training set is not large enough.')

        self.P_y = P_y
        self.G_param = G_param

    def _predict(self, x):
        P_y = self.P_y
        G_param = self.G_param
        pred_P = np.array([])
        c = list(P_y.keys())
        for c_k in P_y.keys():
            p_i_list = []
            for i, xi in enumerate(x):
                mu, sigma = G_param[c_k][i]
                p_i = stats.norm(mu, sigma).pdf(xi)
                p_i_list.append(p_i)
            p = np.prod(P_y[c_k]+p_i_list)
            pred_P = np.append(pred_P, p)
        return c[np.where(pred_P==pred_P.max())[0][0]]

    def predict(self, X):
        if len(X.shape) == 1: X = X.reshape(1, X.shape[0])
        return np.array(list(map(self._predict, X)))

    def score(X, y):
        pass