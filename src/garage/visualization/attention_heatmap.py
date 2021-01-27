import matplotlib
import numpy as np

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # drawing heat map of attention weights

plt.rcParams['font.sans-serif'] = ['SimSun']  # set font family
import matplotlib.ticker as ticker


def plot_attention(data, X_label=None, Y_label=None, title=None, filepath=""):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(20, 8))  # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)

    # Set axis labels
    if X_label is not None and Y_label is not None:

        xticks = range(0, len(X_label))
        centralized_xticks = np.arange(0.5, len(X_label), 1.0)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_locator(ticker.FixedLocator(centralized_xticks))
        ax.xaxis.set_minor_formatter(ticker.FixedFormatter(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks


        yticks = range(0, len(Y_label))
        centralized_yticks = np.arange(0.5, len(Y_label), 1.0)
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(ticker.FixedLocator(centralized_yticks))
        ax.yaxis.set_minor_formatter(ticker.FixedFormatter(Y_label))
        ax.set_yticks(yticks, minor=False)

        ax.grid(True)

    # Save Figure
    plt.title(u'Attention Heatmap:' + str(title))
    file_name = filepath + "/" + str(title) + '.png'
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure