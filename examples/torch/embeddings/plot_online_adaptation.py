import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from os import listdir
from os.path import isfile, join
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

files_path = '/data/transformer-metarl/garage/examples/torch/data/local/results/adaptation'

env_dict = { 'ant-dir': 'AntDir', 'halfcheetah-dir': 'HalfCheetahDir', 'halfcheetah-vel': 'HalfCheetahVel',
'ml45': 'MetaWorld-ML45', 'reach-v2': 'MetaWorld-ML1-Reach-v2', 'push-v2': 'MetaWorld-ML1-Push-v2'}

alg_dict = {'trmrl': 'TrMRL (ours)', 'maml-trpo': 'MAML-TRPO', 'rl2-ppo': 'RL2-PPO', 'pearl': 'PEARL'}

max_timesteps_dict = { 'ant-dir': 50000000, 'halfcheetah-dir': 50000000, 'halfcheetah-vel': 50000000,
'ml45': 50000000, 'reach-v2': 50000000, 'push-v2': 52000000 }

def plot_curve_with_ci(ax, data, x_values, curve_label):
    bs_mean_value = []
    bs_mean_ub = []
    bs_mean_lb = []

    for i in range(data.shape[1]):
        step = data[:, i]
        # mean, lb, ub = mean_confidence_interval(step)
        # bs_mean_value.append(mean)
        # bs_mean_ub.append(min(ub, 1.0))
        # bs_mean_lb.append(max(lb, 0.0))
        bs_mean_step = bs.bootstrap(step, stat_func=bs_stats.mean, alpha=0.05)
        bs_std_step = bs.bootstrap(step, stat_func=bs_stats.std, alpha=0.05)
        bs_mean_value.append(bs_mean_step.value)
        bs_mean_ub.append(bs_mean_step.upper_bound)
        bs_mean_lb.append(bs_mean_step.lower_bound)

    # ax.set_xticks(10 * len(x_values))
    
    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x_values, bs_mean_value, lw = 1, label = curve_label)
    # Shade the confidence interval
    ax.fill_between(x_values, bs_mean_lb, bs_mean_ub,  alpha = 0.4)    


# environments_dirs = ['ant-dir', 'halfcheetah-dir', 'halfcheetah-vel']
environments_dirs = ['halfcheetah-vel']
alg_dirs = ['trmrl']

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.style.use('seaborn')
fig, ax = plt.subplots(1, 2, figsize=(64, 64))
plt_x = 0
plt_y = 0
MAX_EPISODES = 6
MAX_TIMESTEPS = 1200

for env in environments_dirs:
    ax_subplot_timestep = ax[plt_y]
    for alg in alg_dirs:
        runs_dir = join(files_path, env, alg)
        subdirs = [f for f in listdir(runs_dir)]
        env_name = env_dict[env]
        print('There are ' + str(len(subdirs)) + ' experiments in ' + runs_dir)
        test_perf_timestep = []
        max_len = 0
        perf_all_tasks_timestep = []
        for exp in subdirs:
            exp_dir = join(runs_dir, exp)
            files = [f for f in listdir(exp_dir)]
            for i in range(len(files) // 2):   
                perf_task_timestep = pd.read_csv(join(exp_dir, 'rewards_' + str(i) +  '.csv'), engine='python', header=None).values.squeeze()[:MAX_TIMESTEPS]
                perf_all_tasks_timestep.append(perf_task_timestep)

            avg_all_tasks_timestep = np.mean(perf_all_tasks_timestep, axis=0)
            test_perf_timestep.append(avg_all_tasks_timestep)
        
        plot_curve_with_ci(ax_subplot_timestep, np.array(test_perf_timestep), list(range(len(test_perf_timestep[0]))), alg_dict[alg])
    
    # Label the axes and provide a title
    ax_subplot_timestep.set_title(env_name)
    ax_subplot_timestep.set_xlabel('timesteps')
    # ax_subplot.set_ylabel('Success Rate - Train Tasks')
    ax_subplot_timestep.grid()

    # ax_subplot_test.set_title(env_name)
    # ax_subplot_timestep.set_xlabel('timesteps')
    # # ax_subplot_test.set_ylabel('Success Rate - Test Tasks')
    # ax_subplot_timestep.grid()
    # ax_subplot.set_facecolor('#cccccc')

        # fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    plt_y = plt_y + 1

ax[0].set_ylabel('Average Reward')
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='large', shadow=True)
plt.show()
    

# fig.savefig('meta_training_metaworld.svg')  



# for env in 

# for f in files:
    
#     if f.startswith('rewards_stats'):
#         id = f.split('_')[-1].split('.')[0]

#         # Creating dataset
#         data = pd.read_csv(join(files_path, "rewards_" + id + ".csv")).values[:, :200]
#         stats = pd.read_csv(join(files_path, f)).values
#         plot_curve_with_ci(stats, "Online Adaptation", "Episode", "Reward", range(1, stats.shape[1] + 1))
#         plot_curve_with_ci(data, "Online Adaptation - Timesteps", "Timestep", "Reward", range(0, data.shape[1] + 1, 10), x_fontsize=6)