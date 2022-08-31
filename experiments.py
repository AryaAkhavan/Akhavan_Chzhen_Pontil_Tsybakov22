import numpy as np
import logging
import pickle
import os.path
from tqdm import tqdm
from joblib import Parallel, delayed


from plotter import plot_results
from black_box import BlackBox
from oracles import ZeroOrderL1, ZeroOrderL2
from objectives import SumFuncL1Test

N_JOBS = 4
level = logging.INFO
fmt = '[%(levelname)s] %(asctime)s - %(message)s'
logging.basicConfig(level=level, format=fmt)


def inner_loop(i, estimator, sample, setup_optimizer,
             setup_black_box, SIGNATURE, to_cache=True):
    bb = BlackBox(estimator=estimator, **setup_black_box)
    save_file_name = f'cache/{estimator}_{SIGNATURE}{i+1}'
    if to_cache and os.path.isfile(save_file_name):
        logging.debug(f"[{i+1}/{sample}]: loaded")
        with open(save_file_name, "rb") as fp:
            report = pickle.load(fp)
    else:
        logging.debug(f"[{i+1}/{sample}]: optimizing")
        report = bb.optimize(**setup_optimizer)
        with open(save_file_name, "wb") as fp:
            pickle.dump(report, fp)
    return report


def run_loop(estimator, sample, setup_estimator, setup_optimizer,
             setup_black_box, SIGNATURE, to_cache):
    results = Parallel(n_jobs=N_JOBS)(delayed(inner_loop)(i, estimator(**setup_estimator), sample, setup_optimizer, setup_black_box, SIGNATURE, to_cache) for i in tqdm(range(sample)))
    return results


def run_experiment(dim, max_iter, sample, constr_type,
                   radius, objective, norm_str_conv, norm_lipsch,
                   sigma, objective_min, noise_family, to_cache=True):



    sign_arya_style = [to_cache, noise_family, objective_min,
                       sigma, norm_lipsch, norm_str_conv,
                       objective, radius, constr_type, sample,
                       max_iter, dim]


    SIGNATURE = ''.join(f'{value}_'.replace('.', 'DOT') for value in sign_arya_style)



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


    methods = {
    'Our' : ZeroOrderL1,
    'Spherical' : ZeroOrderL2
    }


    results = {}

    for method_name, method in methods.items():

        results[method_name] =  run_loop(method, sample, setup_estimator,
                                         setup_optimizer, setup_black_box,
                                         SIGNATURE, to_cache)

    return results, SIGNATURE


def main(dim, max_iter, sample, constr_type, radius, objective,
         norm_str_conv, norm_lipsch, sigma, objective_min, noise_family,
         to_cache, to_save_plot):
    if not os.path.exists('cache/'):
        os.makedirs('cache/')

    results, SIGNATURE = run_experiment(dim, max_iter, sample,
                                        constr_type, radius,
                                        objective, norm_str_conv,
                                        norm_lipsch, sigma,
                                        objective_min, noise_family,
                                        to_cache)


    return results, SIGNATURE



if __name__ == '__main__':
    dim = 10
    max_iter = 5000
    sample = 20
    constr_type = 'simplex'
    radius = np.log(dim)**(1/2) if constr_type=='simplex' else 1
    objective = SumFuncL1Test(dim=dim)
    norm_str_conv = 1 if constr_type=='simplex' else 2
    norm_lipsch = 1
    sigma = 0.
    objective_min = objective.get_min()
    noise_family = 'Gaussian'
    to_plot = True
    to_cache = True
    to_save_plot = True

    results, SIGNATURE = main(dim, max_iter, sample, constr_type, radius,
                              objective, norm_str_conv, norm_lipsch, sigma,
                              objective_min, noise_family, to_cache,
                              to_save_plot)


    if to_plot:
        plot_results(max_iter, dim, constr_type,
                     objective_min, results, SIGNATURE, to_save_plot)