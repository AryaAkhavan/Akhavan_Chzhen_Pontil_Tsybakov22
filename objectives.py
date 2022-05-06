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

class New_Test():
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
        return "New_test"

class F_Test():
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
        val = abs(x[0] - 1)

        if self.dim > 1:
            for i in range(self.dim - 1):
                val += abs(1 + x[i + 1] - 2 * x[i])
        return val


    def __format__(self, format_spec=None):
        return "F_test"
