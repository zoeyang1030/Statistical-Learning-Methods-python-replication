import numpy as np
import matplotlib.pyplot as plt

class LinearClassifierPlot(object):
    def __init__(self, ax=None):
        if not ax: _, self.ax = plt.subplots(figsize=(8, 6))
        else: self.ax = ax
        self.class_dict = {'class_n':0}

    def _template(self):
        self.ax.set_xlabel("First feature", fontsize=13)
        self.ax.set_ylabel("Second feature", fontsize=13)
        self.ax.set_title('Feature Visualization', weight='bold', fontsize=19)

    def scatter_plot(self, X, y):
        y_source = np.unique(y)
        l = len(y_source)
        marker = ['o', '^', 'v']
        color = ['w', 'gray', 'black']
        for y_label in y_source:
            x = X[y==y_label]
            if y_label in self.class_dict:
                j = self.class_dict[y_label]
            else:
                if self.class_dict['class_n'] > 2:
                    raise ValueError('Max number of classes: 3.')
                self.class_dict[y_label] = self.class_dict['class_n']
                self.class_dict['class_n'] += 1
                j = self.class_dict[y_label]
                self.ax.scatter(x[:, 0], x[:, 1], marker=marker[j], c=color[j],
                                label='Class %s'%j, edgecolor='black', 
                                linewidth=1, s=110, alpha=0.7)
    
    def hyperplane_plot(self, model, label='Hyperplane', c='gray', ls='-'):
        if not self.ax: Exception('Must plot data first.') 
        coef = model.get_coef()
        y = []
        k = - coef[0] / coef[1]
        b = coef[-1]
        for x in self.ax.get_xlim():
            y.append(k*x+b)
        
        self.ax.plot(self.ax.get_xlim(), y, color=c, ls=ls, label=label)

    def clear(self):
        self.ax = None

    def show(self):
        self._template()
        return self.ax.figure