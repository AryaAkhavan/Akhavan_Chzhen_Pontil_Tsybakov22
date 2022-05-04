import numpy as np
from black_box import BlackBox
from oracles import ZeroOrderL1, ZeroOrderL2
import matplotlib.pyplot as plt
from objectives import FuncL2Test, FuncL1Test
from timeit import default_timer as timer
import math
import statistics
import seaborn as sns


def plot_results(max_iter dim, constr_type, mean_l1, std_l1,
                 mean_l2, std_l2, SIGNATURE, to_save=False):


    error_l1 = np.array(stack_l1) - objective_min
    std_l1 = np.array(error_l1).std(0)
    mean_l1 = np.average(error_l1, axis=0)

    error_l2 = np.array(stack_l2) - objective_min
    std_l2 = np.array(error_l2).std(0)
    mean_l2 = np.average(error_l2, axis=0)


    colours = sns.color_palette('colorblind')
    plt.plot(np.arange(max_iter)+1, np.array(mean_l1), color=colours[0],
             label='Our', linestyle='--')
    plt.fill_between(np.arange(max_iter)+1, mean_l1-std_l1, mean_l1+std_l1,
                     facecolor=colours[0], alpha=0.3)
    plt.plot(np.arange(max_iter)+1, np.array(mean_l2), color=colours[3],
             label='Spherical', linestyle='-.')
    plt.fill_between(np.arange(max_iter)+1, mean_l2-std_l2, mean_l2+std_l2,
                     facecolor=colours[3], alpha=0.3)
    font2 = {'family': 'serif', 'color': 'black', 'size': 12}
    plt.xlabel('Number of iterations', fontdict=font2)
    plt.ylabel("Optimization error", fontdict=font2)



    plt.yscale('log')
    plt.xscale('log')
    plt.title(f"Constraint type: {constr_type}, dimension: {dim}")
    plt.legend()


    plt.show()
    if to_save:
        plt.savefig('plots/SIGNATURE.pdf')


if __name__ == '__main__':
    dim = 3
    max_iter = 10000
    sample = 100
    constr_type = 'simplex'
    radius = math.log(dim)**(1/2)
    objective = FuncL1Test(dim=dim)
    norm_str_conv = 1
    norm_lipsch = 1
    to_plot = True
    sigma = 0
    objective_min = objective.get_min()
    noise_family = 'Bernoulli'



    """
        TODO:
            1. Create a unique signature of an experiment.
            2. Create two folders: one for cached experiments + for plots
            3. Cached folder shouldn't be in git, plots can be in git
            4. Before running experiments check if such signature exists in cached folder, if not then run experiments, otherwise load from cache (or make a parameter to_load_from_cache=True/False)
            5. Paralelize the loop for the variance (joblib or numba)
    """

    # SIGNATURE = f"{dim}_{max_iter}_{sample}_{constr_type}_{int(radius)}_{norm_str_conv}_{?}.npy"

    # folder with all the files:
    #     - cached (never in git)
    #     - plots  (can be in git)


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
        print(f"Run [{i+1} / {sample}]: finished in: {(end - start):.2f}sec")

    # for i in range(sample):
    #     if SIGNATURE_file_exists:
    #         # report_l1 = load(SIGNATURE_file)
    #         stack_l1.append(report_l1)
    #         print("Run [{i+1} / {sample}]: loaded from cache.")

    #     else:
    #         start = timer()
    #         report_l1 = bb_l1.optimize(**setup_optimizer)
    #         stack_l1.append(report_l1)
    #         end = timer()
    #         print(f"Run [{i+1} / {sample}]: finished in: {(end - start):.2f}sec")

    stack_l2 = []
    for i in range(sample):
        start = timer()
        report_l2 = bb_l2.optimize(**setup_optimizer)
        stack_l2.append(report_l2)
        end = timer()
        print(f"Run [{i+1} / {sample}] \n Spherical method finished in: {(end - start):.2f}sec")




    if to_plot:
        plot_results(max_iter, dim, constr_type,
                     objective_min, stack_l1,
                     stack_l2, SIGNATURE)