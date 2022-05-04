import numpy as np
from black_box import BlackBox
from oracles import ZeroOrderL1, ZeroOrderL2
import matplotlib.pyplot as plt
from objectives import FuncL2Test, FuncL1Test
from timeit import default_timer as timer
import math
import statistics

if __name__ == '__main__':
    dim = 4000
    max_iter = 10000
    sample = 3
    constr_type = 'simplex'
    radius = math.log(dim)**(1/2)
    objective = FuncL1Test(dim=dim)
    norm_str_conv = 1
    norm_lipsch = 1
    to_plot = True
    sigma = 0
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

    stack_l1 = []
    for i in range(sample):
        start = timer()
        report_l1 = bb_l1.optimize(**setup_optimizer)
        stack_l1.append(report_l1)
        end = timer()
        print(i+1,"l1")
    #print(f"Our method finished in: {(end - start):.2f}sec")
    error_l1 = np.array(stack_l1) - objective_min
    std_l1 = np.array(error_l1).std(0)
    mean_l1 = np.average(error_l1, axis=0)
    s1_l1 = mean_l1 - std_l1
    s2_l1 = mean_l1 + std_l1

    stack_l2 = []
    for i in range(sample):
        start = timer()
        report_l2 = bb_l2.optimize(**setup_optimizer)
        stack_l2.append(report_l2)
        end = timer()
        print(i+1, "l2")
    error_l2 = np.array(stack_l2) - objective_min
    std_l2 = np.array(error_l2).std(0)
    mean_l2 = np.average(error_l2, axis=0)
    s1_l2 = mean_l2 - std_l2
    s2_l2 = mean_l2 + std_l2
    #print(f"Spherical method finished in: {(end - start):.2f}sec")


    if to_plot:
        plt.plot(np.arange(max_iter)+1, np.array(mean_l1), color = 'red',
                 label='Algorithm 1')
        plt.fill_between(np.arange(max_iter)+1, s1_l1, s2_l1, facecolor='khaki', label='d = 100')
        plt.plot(np.arange(max_iter)+1, np.array(mean_l2), color = 'green',
                 label='Spherical')
        plt.fill_between(np.arange(max_iter)+1, s1_l2, s2_l2, facecolor='peachpuff', label='d = 100')
        plt.axhline(y=0, linestyle='-.')
        font2 = {'family': 'serif', 'color': 'darkred', 'size': 12}
        plt.xlabel('Number of iterations', fontdict=font2)
        plt.ylabel('Optimization error (d = 4000)', fontdict=font2)



        plt.yscale('log')
        plt.xscale('log')
        #plt.title(f"Constraint type: {constr_type}, dimension: {dim}")
        plt.legend()
        plt.show()
        plt.figure(dpi=800)