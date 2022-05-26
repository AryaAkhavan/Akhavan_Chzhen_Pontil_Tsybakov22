import numpy as np
from helpers import norm_dual, sample_spherical


class Oracle:
    def estimate(self, x, t, function):
        raise NotImplementedError


class ZeroOrderL1(Oracle):
    def __init__(self, dim=100,
                 radius=1,
                 norm_str_conv=2,
                 norm_lipsch=2):
        """Our proposed method

        Parameters
        ----------
        dim : int, optional
            dimension
        radius : int, optional
            prox radius
        norm_str_conv : int, optional
            strong convexity norm
        norm_lipsch : int, optional
            lipshcitz norm
        """
        self.dim = dim
        self.radius = radius
        self.norm_str_conv = norm_str_conv
        self.norm_lipsch = norm_lipsch


    def __format__(self, format_spec=None):
        if format_spec == 'b':
            return "Our method"
        return "our_method"


    def discretization(self, t, noisy):
        # return self.radius / np.sqrt(t)
        bqd_inv = self.dim + 1
        if self.norm_lipsch < np.log(self.dim):
            bqd_inv /= self.norm_lipsch * self.dim ** (1 / self.norm_lipsch)
        else:
            bqd_inv /= np.exp(1) * np.log(self.dim)
        if not noisy:
            bqd_inv *= self.dim ** (1/2 - 1/self.norm_str_conv + 1/min(2, self.norm_lipsch))
            return self.radius * bqd_inv / (200 * np.sqrt(t))
        else:
            return np.sqrt(1.25 * self.radius * bqd_inv / np.sqrt(t)) * self.dim ** (1 - .5 / self.norm_str_conv)


    def estimate(self, x, t, function, noisy):
        h = self.discretization(t, noisy)
        v = sample_spherical(self.dim, norm_algo=1)
        v2 = np.sign(v)
        x_left = x - h * v
        x_right = x + h * v
        delta = function(x_right, t) - function(x_left, t)
        grad = self.dim * delta * v2 / (2 * h)
        dual_norm = np.abs(grad[0]) ** 2 * self.dim ** (2 / norm_dual(self.norm_str_conv))
        return grad, dual_norm



class ZeroOrderL2(Oracle):
    def __init__(self, dim=100,
                 radius=1,
                 norm_str_conv=2,
                 norm_lipsch=2):
        """Spherical based method

        Parameters
        ----------
        dim : int, optional
            dimension
        radius : int, optional
            prox radius
        norm_str_conv : int, optional
            strong convexity norm
        norm_lipsch : int, optional
            lipshcitz norm
        """
        self.dim = dim
        self.radius = radius
        self.norm_str_conv = norm_str_conv
        self.norm_lipsch = norm_lipsch


    def __format__(self, format_spec=None):
        if format_spec == 'b':
            return "L2 Spherical method"
        return "spherical_method"


    def discretization(self, t, noisy):
        if not noisy:
            return self.radius / np.sqrt(t)
        else: return np.sqrt(self.radius / np.sqrt(t))


    def estimate(self, x, t, function, noisy):
        h = self.discretization(t, noisy)
        v = sample_spherical(self.dim, norm_algo=2)
        x_left = x - h * v
        x_right = x + h * v
        delta = function(x_right, t) - function(x_left, t)
        grad = self.dim * delta * v / (2 * h)
        dual_norm = np.linalg.norm(grad, ord=norm_dual(self.norm_str_conv)) ** 2
        return grad, dual_norm




