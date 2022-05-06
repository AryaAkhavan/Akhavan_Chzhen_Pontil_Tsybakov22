import numpy as np
from helpers import softmax


class FuncL2Test():
    def __init__(self, dim=5, center=None):
        self.dim = dim
        if center is None:
            self.center = np.ones(dim) / (4 * np.sqrt(dim))
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
        return np.linalg.norm(x - self.center) ** 2


    def __format__(self, format_spec=None):
        return "L2_test_function"


class FuncL1Test():
    def __init__(self, dim=5, center=None):
        """

        :rtype: object
        """
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


    def __format__(self, format_spec=None):
        return "L1_test_function"

class NewTest():
    def __init__(self, dim=5, center=None):
        """

        :rtype: object
        """
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
        return np.linalg.norm(x - self.center) ** 2 + np.linalg.norm(x - 0.1*self.center, ord=1)


    def __format__(self, format_spec=None):
        return "NewTest"

class FTest():
    def __init__(self, dim=5, center=None):
        """

        :rtype: object
        """
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
        return np.abs(x[0]-1) + np.linalg.norm(1 +x[2:] -2*x[1:self.dim-1], ord=1)


    def __format__(self, format_spec=None):
        return "FTest"
