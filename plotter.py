import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def plot_results(max_iter, dim, constr_type,
                 objective_min, results, SIGNATURE, to_save=False):

    grid = np.arange(max_iter)+1

    jump_default = 100
    multiplier = 50

    jump = jump_default if multiplier * max_iter > jump_default else 1
    colours = sns.color_palette('colorblind')

    labels = {
    'Our' : r'\Large $\ell_1$-randomization (Our)',
    'Spherical' : r'\Large $\ell_2$-randomization',
    'Gaussian' : r'\Large Gaussian randomization',
    }

    lines = {
    'Our' : '-',
    'Spherical' : '-.',
    'Gaussian' : '--',
    }


    plt.figure(figsize=(7,3))
    for idx, (method_name, result) in enumerate(results.items()):

        error = np.array(result) - objective_min
        std = np.array(result).std(0)
        mean = np.average(result, axis=0)



        plt.plot(grid[0::jump], np.array(mean)[0::jump], color=colours[idx],
                 label=labels[method_name], linestyle=lines[method_name])
        plt.fill_between(grid[0::jump], mean[0::jump]-std[0::jump],
                         mean[0::jump]+std[0::jump],
                         facecolor=colours[idx], alpha=0.3)


    plt.xlabel(r'\Large Number of iterations')
    plt.ylabel(r"\Large Function value")



    plt.yscale('log')
    #plt.xscale('log')
    plt.title(rf"\Large Constraint type: {{\Large\texttt {constr_type}}}; \Large dimension: \Large $d={dim}$")
    plt.legend()


    if to_save:
        if not os.path.exists('plots/'):
            os.makedirs('plots/')
        plt.savefig(f'plots/{SIGNATURE}.pdf',  bbox_inches='tight')
    plt.show()