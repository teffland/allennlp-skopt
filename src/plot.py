import os
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

def plot_results(base_dir, results):
    """ Plot convergence, evaluations, and surrogate objectives to files.
    """
    ax = plot_convergence(results)
    plt.savefig(os.path.join(base_dir,'convergence.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(base_dir,'convergence.png'), bbox_inches='tight')
    plt.close()

    ax = plot_evaluations(results)
    reformat_plot(ax)
    plt.savefig(os.path.join(base_dir,'evaluations.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(base_dir,'evaluations.png'), bbox_inches='tight')
    plt.close()

    ax = plot_objective(results, sample_source='result')
    reformat_plot(ax)
    plt.savefig(os.path.join(base_dir,'surrogate_objective.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(base_dir,'surrogate_objective.png'), bbox_inches='tight')
    plt.close()



def reformat_plot(axs):
    """ Reformat skopt plots to cope with long dimension names.
    """
    n = len(axs)
    for i in range(len(axs)):
        # Make all the left column axis y labels multiline horizontal right aligned
        if i > 0:
            ax = axs[i,0]
            ax.set_ylabel(clean_nested_key(ax.get_ylabel()).replace('->','\n.'), rotation=0, ha='right')

        # Make all the bottom row axis x labels multiline horizontal center aligned
        ax = axs[n-1,i]
        ax.set_xlabel(clean_nested_key(ax.get_xlabel()).replace('->','\n.'), rotation=0, ha='center')

        # Make all diagonal axis x labels multiline horizontal right aligned
        ax = axs[i,i]
        ax.set_xlabel(clean_nested_key(ax.get_xlabel()).replace('->','\n.'), rotation=0, ha='left')
