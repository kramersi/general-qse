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

    return np.sum(err)/err.size


def cosine_diff(x1, x2, axis=0, plot=False):

    s_up = np.sum(x1*x2, axis=axis)
    s1 = np.sqrt(np.sum(x1**2, axis=axis))
    s2 = np.sqrt(np.sum(x2**2, axis=axis))
    s = s_up / (s1 * s2)

    if plot:
        plt.figure(3)
        pd.DataFrame(1-s).plot()
        plt.show()

    print('avg cosine diff: ', np.mean(1-s))

    return np.mean(1 - s)


def cross_corr(x1, x2, col=4):
    corr_list = []
    if col > 1:
        for i in range(col):
            sc = np.corrcoef(x1[:, i], x2[:, i])
            corr_list.append(sc[0, 1])
        corr = np.mean(corr_list)
    else:
        corr = np.corrcoef(x1, x2)[0, 1]

    print('crosscorr: ', corr)

    return corr


def spearman_corr(x1, x2, axis=0, col=4):

    s_corr = spearmanr(x1, x2, axis=axis)[0]
    s_tot = np.array([s_corr[i, i + col] for i in range(int(s_corr.shape[0]/2))])
    s_mean = np.mean(abs(s_tot))

    print('avg spearman correlation: ', s_mean)

    return s_mean

