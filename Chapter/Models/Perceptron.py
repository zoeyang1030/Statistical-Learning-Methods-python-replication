import numpy as np

class PerceptronLinearAlgorithm(object):
    def __init__(self, lr=0.01, max_iter=1e5, model_type='traditional', margin=None):
        self._w = None
        self._b = None
        self._c = None
        self._param = {
            'lr': lr, 
            'max_iter': max_iter,
            'model_type': model_type
        }
        if not margin is None: self._param['m'] = margin

    def _perceptron_traditional(self, X, y):
        iteration = 0
        w = np.zeros(X.shape[0]).reshape(1, X.shape[0])
        b = 0.1
        sample = range(X.shape[1])

        while iteration < self._param['max_iter']:
            i = np.random.choice(sample, 1)[0]
            iteration += 1
            if (np.dot(w, X[:, i]) + b) * y[i] <= 0:
                w += self._param['lr'] * y[i] * X[:, i]
                b += self._param['lr'] * y[i]

        self._submit(w[0], b)

    def _perceptron_dualform(self, X, y):
        iteration = 0
        a = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
        b = 0.1
        gram_matrix = np.dot(X.T, X)
        sample = range(X.shape[1])

        while iteration < self._param['max_iter']:
            iteration += 1
            i = np.random.choice(sample, 1)[0]
            xx = gram_matrix[i].reshape(1, X.shape[1])
            ay = a*y.reshape(X.shape[1], 1)
            if (np.dot(xx, ay) + b) * y[i] <= 0:
                a[i] += self._param['lr']
                b += self._param['lr'] * y[i]

        w = np.dot(a.T*y, X.T)
        self._submit(w[0], b)

    def _pocket_algorithm(self, X, y):
        iteration = 0
        w_p = np.zeros(X.shape[0]).reshape(1, X.shape[0])
        b_p = 0.1
        w = np.zeros(X.shape[0]).reshape(1, X.shape[0])
        b = 0
        cor_p = cor = 0
        sample = range(X.shape[1])

        while iteration < self._param['max_iter']:
            iteration += 1
            i = np.random.choice(sample, 1)[0]
            if (np.dot(w_p, X[:, i]) + b_p) * y[i] > 0:
                pred = ((np.dot(w_p, X)+b_p>=0)*2-1)
                cor_p = (pred==y).sum()
                if cor_p > cor:
                    w, b, cor = w_p, b_p, cor_p
                    if cor == X.shape[1]: break
            else:
                w_p += self._param['lr'] * y[i] * X[:, i]
                b_p += self._param['lr'] * y[i]

        self._submit(w[0], b)

    def _perceptron_voted(self, X, y):
        iteration = 0
        k = 0
        sample = X.shape[1]
        w_array = np.array(np.zeros(X.shape[0]).reshape(1, X.shape[0]))
        b_array = c_array = np.zeros(1)

        while iteration < 1000:#self._param['max_iter']:
            for i in range(sample):
                iteration += 1
                if (np.dot(w_array[k], X[:, i]) + b_array[k]) * y[i] > 0:
                    c_array[k] += 1
                else:
                    w_k = w_array[k] + self._param['lr']*y[i]*X[:, i]
                    w_array = np.concatenate([w_array, w_k.reshape(1, X.shape[0])])
                    b_array = np.append(b_array, b_array[k]+self._param['lr']*y[i])
                    c_array = np.append(c_array, 1)
                    k += 1
        
        self._w = w_array
        self._b = b_array
        self._c = c_array

    def _perceptron_averaged(self, X, y):
        sample = X.shape[1]
        w = u = np.zeros(X.shape[0]).reshape(1, X.shape[0])
        b = beta = 0
        c = 0
        iteration = 0

        while iteration < self._param['max_iter']:
            for i in range(sample):
                iteration += 1
                if (np.dot(w, X[:, i]) + b) * y[i] > 0:
                    c += 1
                else:
                    w += self._param['lr']*y[i]*X[:, i]
                    b += self._param['lr']*y[i]
                    u += c*y[i]*X[:, i]
                    beta += c*y[i]

        self._submit((w-u/c)[0], b-beta/c)

    def _perceptron_margin(self, X, y):
        if 'm' in self._param: m = self._param['m']
        else: raise ValueError('m must be not None.')
        iteration = 0
        w = np.zeros(X.shape[0]).reshape(1, X.shape[0])
        b = 0.1
        sample = range(X.shape[1])

        while iteration < self._param['max_iter']:
            i = np.random.choice(sample, 1)[0]
            iteration += 1
            if (np.dot(w, X[:, i]) + b) * y[i] <= -m:
                w += self._param['lr'] * y[i] * X[:, i]
                b += self._param['lr'] * y[i]

        self._submit(w[0], b)

    def _submit(self, w, b):
        self._w = w
        self._b = b

    def fit(self, X, y):
        X = X.T
        if self._param['model_type'] == 'traditional': self._perceptron_traditional(X, y)
        elif self._param['model_type'] == 'dualform': self._perceptron_dualform(X, y)
        elif self._param['model_type'] == 'pocket': self._pocket_algorithm(X, y)
        elif self._param['model_type'] == 'voted': self._perceptron_voted(X, y)
        elif self._param['model_type'] == 'averaged': self._perceptron_averaged(X, y)
        elif self._param['model_type'] == 'margin': self._perceptron_margin(X, y)
        else: raise ValueError('Wrong model_type: %s'%self._param['model_type'])

    def predict(self, X):
        X = X.T
        if self._param['model_type'] == 'voted':
            k = len(self._b)
            s = self._c.reshape(k, 1)*((np.dot(self._w, X)+self._b.reshape(k, 1)>=0)*2-1)
            return (s.sum(0)>=0)*2-1
        else:
            return (np.dot(self._w, X)+self._b>=0)*2-1

    def score(self, X, y):
        return (self.predict(X)==y).sum()/X.shape[0]

    def get_param(self):
        return self._param
    
    def get_coef(self):
        return [i for i in [self._w, self._b, self._c] if not i is None]