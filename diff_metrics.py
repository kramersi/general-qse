import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.stats import spearmanr


def square_diff(x1, x2, plot=False):

    err = (x1 - x2)**2

    if plot:
        plt.figure(2)
        pd.DataFrame(err).plot()
        plt.show()

    print('avg square diff: ', np.sum(err)/err.size)

    return np.sum(err)


def cosine_diff(x1, x2, plot=False):

    s_up = np.sum(x1*x2, axis=1)
    s1 = np.sqrt(np.sum(x1**2, axis=1))
    s2 = np.sqrt(np.sum(x2**2, axis=1))
    s = s_up / (s1 * s2)

    if plot:
        plt.figure(3)
        pd.DataFrame(1-s).plot()
        plt.show()

    print('avg cosine diff: ', np.mean(1-s))

    return np.sum(1 - s)


def spearman_diff(x1, x2, plot=False):
    s_corr = spearmanr(x1, x2, axis=0)[0]
    s_tot = np.mean(s_corr)

    print('avg spearman diff: ', 1 - s_tot)

    return 1 - s_tot

