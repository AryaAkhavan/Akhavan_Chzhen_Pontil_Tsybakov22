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



if __name__ == '__main__':
    dim = 3
    max_iter = 15000
    objective = func_l2_test
    radius = 1
    norm_str_conv = 2
    constr_type = 2
    norm_lipsch = 2
    to_plot = True
    sigma = 1
    objective_min = 0


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
        plt.plot(np.arange(max_iter) + 1, report_l1, label='Our')
        plt.plot(np.arange(max_iter) + 1, report_l2, label='Spherical')
        plt.axhline(y=objective_min, color='r', linestyle='-.')
        plt.xlabel('Number of iterations')
        plt.ylabel('Objective function')
        plt.title(f"Constraint type: {constr_type}, dimension: {dim}")
        plt.legend()
        plt.show()