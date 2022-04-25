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
        if self.min is not None:
            self.min = 0
            return self.min
        else:
            return self.min__


    def eval(self, x):
        return np.linalg.norm(x - self.center) ** 2



def func_l2_test(x):
    d = len(x)
    center = np.ones(d) / (4 * np.sqrt(d))
    return np.linalg.norm(x - center) ** 2


def func_l1_test(x):
    d = len(x)
    return np.linalg.norm(x - softmax(np.arange(d)), ord=1)


def func_l1_corner_test(x):
    d = len(x)
    center = np.zeros(d)
    center[0] = 1
    return np.linalg.norm(x - center, ord=1)


def func_l1_center_test(x):
    d = len(x)
    center = np.ones(d) / d
    return np.linalg.norm(x - center, ord=1)
