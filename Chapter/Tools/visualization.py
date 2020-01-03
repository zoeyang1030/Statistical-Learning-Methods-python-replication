import numpy as np
import matplotlib.pyplot as plt

class LinearClassifierPlot(object):
    def __init__(self, title='Feature Visualization', ax=None):
        self.title = title
        if not ax: 
            _, self.ax = plt.subplots(figsize=(4.5, 5))
            self._template()
        else: 
            self.ax = ax
            self._template()
        self.class_dict = {'class_n':0}
        self.lim = {}

    def _template(self):
        self.ax.set_xlabel("First feature", fontsize=11)
        self.ax.set_ylabel("Second feature", fontsize=11)
        self.ax.set_title(self.title, weight='bold', fontsize=14)

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
                                linewidth=1, s=80, alpha=0.7)

        self.lim['x_min'] = min(self.ax.get_xlim())
        self.lim['x_max'] = max(self.ax.get_xlim())
        self.lim['y_min'] = min(self.ax.get_ylim())
        self.lim['y_max'] = max(self.ax.get_ylim())
        
        self.ax.legend(loc=2)
    
    def hyperplane_plot(self, model, c='black', ls='--'):
        if not self.ax: Exception('Must plot data first.') 
        xx, yy = np.meshgrid(np.arange(self.lim['x_min'], self.lim['x_max'], 0.02),
                             np.arange(self.lim['y_min'], self.lim['y_max'], 0.02))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        self.ax.contour(xx, yy, Z, colors=c, linestyles=ls, alpha=0.5)
        self.ax.set_xlim(self.lim['x_min'], self.lim['x_max'])
        self.ax.set_ylim(self.lim['y_min'], self.lim['y_max'])
        self.ax.legend(loc=2)

    def clear(self):
        self.ax = None

    def show(self):
        self._template()
        return self.ax.figure