import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self,learning_rate=0.01,n_iter=2000):
        self.weight=0
        self.bias=0
        self.learning_rate=learning_rate
        self.n_iter=n_iter
        self.mean=0
        self.std=0

    def scale(self,x):
        self.mean=np.mean(x)
        self.std=np.std(x)
        return (x-self.mean)/self.std
    
    def fit(self,x,y):
        x=self.scale(x)
        n_sample,n_feat=np.shape(x)
        len=len(x)
        self.weight=np.zeros(n_feat)
        for i in range(self.n_iter):
            y_pred=self.weight@x+self.bias
            error=y-y_pred
            dw=(-2/n_sample)*np.dot(x.T,error)
            db=(-2/n_sample)*np.sum(error)
            self.weight-=self.learning_rate*dw
            self.bias-=self.learning_rate*db

    def predict(self,x):
        x=(x-self.mean)/self.std
        return self.weight@x+self.bias
    
    def save(self, filename):

        np.savez(
            filename,
            weights=self.weight,
            bias=self.bias,
            mean=self.mean,
            std=self.std

        )

    def load(self, filename):
        data = np.load(filename)
        self.weight = data['weights']
        self.bias = data['bias']
        self.mean = data['mean']
        self.std = data['std']
