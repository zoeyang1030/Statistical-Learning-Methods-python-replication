import numpy as np

class Perceptron(object):
    def __init__(self, lr=0.01, max_iter=1e5):
        self._w = None
        self._b = None
        self._param = {
            'lr': lr, 
            'max_iter': max_iter
        }

    def _perceptron_traditional(self, X, y):
        iteration = 0
        w = np.zeros(X.shape[1])
        b = 0
        while iteration < self._param['max_iter']:
            all_correct = 1
            for i in range(X.shape[0]):
                iteration += 1
                if (np.dot(X[i], w) + b) * y[i] <= 0:
                    w += self._param['lr'] * y[i] * X[i]
                    b += self._param['lr'] * y[i]
                    all_correct = 0
            if all_correct: break

        self._standardization(w, b)

    def _perceptron_dualform(self, X, y):
        iteration = 0
        a = np.zeros(X.shape[0])
        b = 0
        gram_matrix = np.dot(X, X.T)

        while iteration < self._param['max_iter']:
            all_correct = 1
            for i in range(X.shape[0]):
                iteration += 1
                if (np.dot(a*y, gram_matrix[:, i]) + b) * y[i] <= 0:
                    a[i] += self._param['lr']
                    b += self._param['lr'] * y[i]
                    all_correct = 0
            if all_correct: break

        w = (X*(y*a).reshape(X.shape[0],1)).sum(axis=0)

        self._standardization(w, b)

    def _standardization(self, w, b):
        w2 = (w**2).sum()**(1/2)
        self._w = w / w2
        self._b = b / w2

    def fit(self, X, y, model_type=0):
        if model_type == 0: self._perceptron_traditional(X, y)
        elif model_type == 1: self._perceptron_dualform(X, y)
        else: raise ValueError('Wrong model_type: %s'%model_type)

    def predict(self, X):
        return (np.dot(X, self._w) + self._b >= 0) * 2 - 1

    def score(self, X_test, y_test):
        return (self.predict(X_test) == y_test).sum() / X_test.shape[0]

    def get_param(self):
        return self._param
    
    def get_coef(self):
        return np.append(self._w, self._b)