import numpy as np
from black_box import BlackBox
from oracles import ZeroOrderL1, ZeroOrderL2
import matplotlib.pyplot as plt
from objectives import FuncL2Test, FuncL1Test, FuncL1Test_ex2
from timeit import default_timer as timer


if __name__ == '__main__':
    dim = 50
    max_iter = 100000
    sample = 10
    constr_type = 2
    radius = 1
    objective = FuncL1Test(dim=dim)
    norm_str_conv = 2
    norm_lipsch = 2
    to_plot = True
    sigma = 1
    objective_min = objective.get_min()
    noise_family = 'Bernoulli'


    black_box_setup = {
    'objective': objective,
    'sigma': sigma,
    'noise_family': noise_family
    }


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

    bb_l1 = BlackBox(estimator=estimator_l1, **black_box_setup)
    bb_l2 = BlackBox(estimator=estimator_l2, **black_box_setup)

    average_l1 = []
    for i in range(sample):
        start = timer()
        report_l1 = bb_l1.optimize(**setup_optimizer)
        average_l1.append(report_l1)
        end = timer()
    #print(f"Our method finished in: {(end - start):.2f}sec")
    average_l1 = np.average(average_l1, axis = 0)

    average_l2 = []
    for i in range(sample):
        start = timer()
        report_l2 = bb_l2.optimize(**setup_optimizer)
        average_l2.append(report_l2)
        end = timer()

    average_l2 = np.average(average_l2, axis=0)
    #print(f"Spherical method finished in: {(end - start):.2f}sec")

    if to_plot:
        plt.plot(np.arange(max_iter)+1, np.array(average_l1)-objective_min,
                 label='Our')
        plt.plot(np.arange(max_iter)+1, np.array(average_l2)-objective_min,
                 label='Spherical')
        plt.axhline(y=0, color='b', linestyle='-.')
        plt.xlabel('Number of iterations')
        plt.ylabel('Optimization error')
        plt.yscale('log')
        plt.xscale('log')
        #plt.title(f"Constraint type: {constr_type}, dimension: {dim}")
        plt.legend()
        plt.show()