import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from os import listdir
from os.path import isfile, join
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

files_path = '/data/transformer-metarl/garage/examples/torch/embeddings'

files = [f for f in listdir(files_path) if isfile(join(files_path, f))]

def plot_curve_with_ci(data, title, xlabel, ylabel, xticks, x_fontsize=12):
    plt.figure(figsize=[6, 6])
    fig, ax = plt.subplots()

    bs_mean_value = []
    bs_mean_ub = []
    bs_mean_lb = []

    for i in range(data.shape[1]):
        step = data[:, i]
        bs_mean_step = bs.bootstrap(step, stat_func=bs_stats.mean, alpha=0.05)
        bs_std_step = bs.bootstrap(step, stat_func=bs_stats.std, alpha=0.05)
        bs_mean_value.append(bs_mean_step.value)
        bs_mean_ub.append(bs_mean_step.upper_bound)
        bs_mean_lb.append(bs_mean_step.lower_bound)

    plt.xticks(xticks, fontsize=x_fontsize)
    
    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(range(1, len(bs_mean_value) + 1), bs_mean_value, lw = 1, label = id)
    # Shade the confidence interval
    ax.fill_between(range(1, len(bs_mean_value) + 1), bs_mean_lb, bs_mean_ub,  alpha = 0.4)
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Display legend
    ax.legend(loc = 'best')

    ax.grid()
    fig.savefig(title + ".svg")    




for f in files:
    
    if f.startswith('rewards_stats'):
        id = f.split('_')[-1].split('.')[0]

        # Creating dataset
        data = pd.read_csv(join(files_path, "rewards_" + id + ".csv")).values[:, :200]
        stats = pd.read_csv(join(files_path, f)).values
        plot_curve_with_ci(stats, "Online Adaptation", "Episode", "Reward", range(1, stats.shape[1] + 1))
        plot_curve_with_ci(data, "Online Adaptation - Timesteps", "Timestep", "Reward", range(0, data.shape[1] + 1, 10), x_fontsize=6)