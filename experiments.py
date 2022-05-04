import numpy as np
from black_box import BlackBox
from oracles import ZeroOrderL1, ZeroOrderL2
from objectives import FuncL2Test, FuncL1Test
import time
import math
import statistics
import logging
import pickle
import os.path
from tqdm import tqdm
from plotter import plot_results
from joblib import Parallel, delayed

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

    return stack_l1, stack_l2, SIGNATURE


if __name__ == '__main__':
    level = logging.INFO
    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=level, format=fmt)


    dim = 3000
    max_iter = 5000
    sample = 4
    constr_type = 'simplex'
    radius = math.log(dim)**(1/2)
    objective = FuncL1Test(dim=dim)
    norm_str_conv = 1
    norm_lipsch = 1
    sigma = 0
    objective_min = objective.get_min()
    noise_family = 'Bernoulli'
    to_plot = True


    if not os.path.exists('cache/'):
        os.makedirs('cache/')

    stack_l1, stack_l2, SIGNATURE = run_experiment(dim, max_iter, sample,
                                                   constr_type, radius,
                                                   objective, norm_str_conv,
                                                   norm_lipsch, sigma,
                                                   objective_min, noise_family)

    if to_plot:
        plot_results(max_iter, dim, constr_type,
                     objective_min, stack_l1,
                     stack_l2, SIGNATURE, True)