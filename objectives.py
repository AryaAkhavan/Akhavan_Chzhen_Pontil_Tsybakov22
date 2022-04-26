import numpy as np
from helpers import softmax


class FuncL2Test():
    def __init__(self, dim=5, center=None):
        self.dim = dim
        if center is None:
            self.center = np.ones(dim) / (4 * np.sqrt(dim))
        else:
            self.center = center
        self.min__ = None


    def get_min(self):
        if self.min_ is None:
            self.min_ = 0.
            return self.min_
        else:
            return self.min_


    def eval(self, x):
        return np.linalg.norm(x - self.center) ** 2


class FuncL1Test():
    def __init__(self, dim=5, center=None):
        self.dim = dim
        if center is None:
            self.center = softmax(np.arange(dim))
        else:
            self.center = center
        self.min_ = None


    def get_min(self):
        if self.min_ is None:
            self.min_ = 0.
            return self.min_
        else:
            return self.min_


    def eval(self, x):
        return np.linalg.norm(x - self.center, ord=1)
