import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_results(max_iter, dim, constr_type,
                 objective_min, results, SIGNATURE, to_save=False):

    grid = np.arange(max_iter)+1

    jump_default = 100
    multiplier = 50

    jump = jump_default if multiplier * max_iter > jump_default else 1
    colours = sns.color_palette('colorblind')



    for idx, (method_name, result) in enumerate(results.items()):

        error = np.array(result) - objective_min
        std = np.array(result).std(0)
        mean = np.average(result, axis=0)



        plt.plot(grid[0::jump], np.array(mean)[0::jump], color=colours[idx],
                 label=method_name, linestyle='--')
        plt.fill_between(grid[0::jump], mean[0::jump]-std[0::jump],
                         mean[0::jump]+std[0::jump],
                         facecolor=colours[idx], alpha=0.3)



    font2 = {'family': 'serif', 'color': 'black', 'size': 12}
    plt.xlabel('Number of iterations', fontdict=font2)
    plt.ylabel("Function value", fontdict=font2)



    plt.yscale('log')
    #plt.xscale('log')
    plt.title(f"Constraint type: {constr_type}; dimension: {dim}")
    plt.legend()


    if to_save:
        if not os.path.exists('plots/'):
            os.makedirs('plots/')
        plt.savefig(f'plots/{SIGNATURE}.pdf',  bbox_inches='tight')
    plt.show()