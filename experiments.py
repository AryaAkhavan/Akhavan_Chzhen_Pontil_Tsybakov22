import numpy as np
from black_box import BlackBox
from oracles import ZeroOrderL1, ZeroOrderL2
import matplotlib.pyplot as plt
from objectives import FuncL2Test, FuncL1Test
import time
import math
import statistics
import seaborn as sns
import logging
import pickle
import os.path
from tqdm import tqdm
from joblib import Parallel, delayed


def plot_results(max_iter, dim, constr_type,
                 objective_min, stack_l1,
                 stack_l2, to_save=False):

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


def inner_loop(i, estimator, sample, setup_optimizer,
             setup_black_box, SIGNATURE):
    bb = BlackBox(estimator=estimator, **setup_black_box)
    save_file_name = f'cache/{estimator}_{SIGNATURE}{i+1}'
    if os.path.isfile(save_file_name):
        logging.debug(f"[{i+1}/{sample}]: loaded")
        with open(save_file_name, "rb") as fp:
            report = pickle.load(fp)
    else:
        logging.debug(f"[{i+1}/{sample}]: optimizing")
        report = bb.optimize(**setup_optimizer)
        with open(save_file_name, "wb") as fp:
            pickle.dump(report, fp)
    return report


def run_loop(estimator, sample, setup_optimizer,
             setup_black_box, SIGNATURE):
    results = Parallel(n_jobs=4)(delayed(inner_loop)(i, estimator, sample, setup_optimizer, setup_black_box, SIGNATURE) for i in tqdm(range(sample)))
    return results


def run_experiment(dim, max_iter, sample, constr_type,
                   radius, objective, norm_str_conv, norm_lipsch,
                   sigma, objective_min, noise_family):

    SIGNATURE = ''.join(f'{value}_'.replace('.', 'DOT') for key, value in locals().items())



    setup_black_box = {
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


    logging.info(f"Our method started")
    stack_l1 = run_loop(estimator_l1, sample, setup_optimizer,
                        setup_black_box, SIGNATURE)


    logging.info(f"Spherical method started")
    stack_l2 = run_loop(estimator_l2, sample, setup_optimizer,
                        setup_black_box, SIGNATURE)

    return stack_l1, stack_l2


if __name__ == '__main__':
    level = logging.INFO
    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=level, format=fmt)


    dim = 3
    max_iter = 5000
    sample = 40
    constr_type = 'simplex'
    radius = math.log(dim)**(1/2)
    objective = FuncL1Test(dim=dim)
    norm_str_conv = 1
    norm_lipsch = 1
    sigma = 0
    objective_min = objective.get_min()
    noise_family = 'Bernoulli'
    to_plot = True

    """
        TODO:
            5. Paralelize the loop for the variance (joblib or numba)
    """
    if not os.path.exists('cache/'):
        os.makedirs('cache/')

    stack_l1, stack_l2 = run_experiment(dim, max_iter, sample, constr_type,
                   radius, objective, norm_str_conv, norm_lipsch,
                   sigma, objective_min, noise_family)

    if to_plot:
        plot_results(max_iter, dim, constr_type,
                     objective_min, stack_l1,
                     stack_l2)