import numpy as np


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)
    return f_x


def sample_spherical(dim=3, norm_algo=1):
    if norm_algo == 2:
        vec = np.random.randn(dim)
        vec /= np.linalg.norm(vec)
    elif norm_algo == 1:
        vec = np.random.laplace(0, 1, dim)
        vec /= np.linalg.norm(vec, ord=1)
    return vec


def mirror_projection(x, constr_type=2):
    if constr_type == 'euclid_ball':
        cur_norm = np.linalg.norm(x)
        if cur_norm > 1:
             return x / np.linalg.norm(x)
        else: return x

    if constr_type == 'pos':
        x[x < 0] = 0
        if np.linalg.norm(x)==0:
            return x
        return x / np.linalg.norm(x)

    if constr_type == 'simplex':
        return softmax(x)


def norm_dual(order):
    if order == 1:
        return np.inf
    if order == np.inf:
        return 1
    return order / (order - 1)
