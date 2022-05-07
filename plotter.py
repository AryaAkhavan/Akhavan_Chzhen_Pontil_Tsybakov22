import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_results(max_iter, dim, constr_type,
                 objective_min, stack_l1,
                 stack_l2, SIGNATURE, to_save=False):

    range = np.arange(max_iter)+1
    error_l1 = np.array(stack_l1) - objective_min
    std_l1 = np.array(error_l1).std(0)
    mean_l1 = np.average(error_l1, axis=0)

    error_l2 = np.array(stack_l2) - objective_min
    std_l2 = np.array(error_l2).std(0)
    mean_l2 = np.average(error_l2, axis=0)

    if (max_iter) > 10**5:
        jumps = np.arange(0, max_iter, step=10**3)
        std_l1 = std_l1[jumps[jumps < max_iter]]
        mean_l1 = mean_l1[jumps[jumps < max_iter]]

        std_l2 = std_l2 [jumps[jumps < max_iter]]
        mean_l2 = mean_l2[jumps[jumps < max_iter]]

        range = range[jumps[jumps < max_iter]]+1

    colours = sns.color_palette('colorblind')
    plt.plot(range, np.array(mean_l1), color=colours[0],
             label='Our', linestyle='--')
    plt.fill_between(range, mean_l1-std_l1, mean_l1+std_l1,
                     facecolor=colours[0], alpha=0.3)


    plt.plot(range, np.array(mean_l2), color=colours[3],
             label='Spherical', linestyle='-.')
    plt.fill_between(range, mean_l2-std_l2, mean_l2+std_l2,
                     facecolor=colours[3], alpha=0.3)


    font2 = {'family': 'serif', 'color': 'black', 'size': 12}
    plt.xlabel('Number of iterations', fontdict=font2)
    plt.ylabel("Optimization error", fontdict=font2)



    plt.yscale('log')
    #plt.xscale('log')
    plt.title(f"Constraint type: {constr_type}, dimension: {dim}")
    plt.legend()


    if to_save:
        if not os.path.exists('plots/'):
            os.makedirs('plots/')
        plt.savefig(f'plots/{SIGNATURE}.pdf',  bbox_inches='tight')
    plt.show()