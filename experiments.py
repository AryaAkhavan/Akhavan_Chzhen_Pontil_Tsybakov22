import numpy as np
from black_box import BlackBox
from oracles import ZeroOrderL1, ZeroOrderL2
import matplotlib.pyplot as plt
from helpers import softmax
from timeit import default_timer as timer


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


if __name__ == '__main__':
    dim = 5
    max_iter = 10000
    objective = func_l1_corner_test
    radius = np.sqrt(np.log(dim))
    norm_str_conv = 1
    constr_type = 'simplex'
    norm_lipsch = 1
    to_plot = True
    sigma = 0.5
    objective_min = 0.


    setup_estimator = {
    'dim': dim,
    'norm_lipsch': norm_lipsch,
    'norm_str_conv': norm_str_conv,
    'radius': radius
    }

    setup_optimizer = {
    'max_iter': max_iter,
    'constr_type': constr_type,
    'radius': radius,
    'norm_str_conv': norm_str_conv
    }

    estimator_l1 = ZeroOrderL1(**setup_estimator)
    estimator_l2 = ZeroOrderL2(**setup_estimator)

    bb_l1 = BlackBox(estimator=estimator_l1, objective=objective, sigma=sigma)
    bb_l2 = BlackBox(estimator=estimator_l2, objective=objective, sigma=sigma)

    start = timer()
    report_l1 = bb_l1.optimize(**setup_optimizer)
    end = timer()
    print(f"Our method finished in: {(end - start):.2f}sec")

    start = timer()
    report_l2 = bb_l2.optimize(**setup_optimizer)
    end = timer()
    print(f"Spherical method finished in: {(end - start):.2f}sec")

    if to_plot:
        plt.plot(np.arange(max_iter)+1, np.array(report_l1)-objective_min,
                 label='Our')
        plt.plot(np.arange(max_iter)+1, np.array(report_l2)-objective_min,
                 label='Spherical')
        plt.axhline(y=0, color='b', linestyle='-.')
        plt.xlabel('Number of iterations')
        plt.ylabel('Optimization error')
        plt.yscale('log')
        plt.xscale('log')
        plt.title(f"Constraint type: {constr_type}, dimension: {dim}")
        plt.legend()
        plt.show()