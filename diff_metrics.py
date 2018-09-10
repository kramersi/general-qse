import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.stats import spearmanr
from os import listdir

# -------------------------------------------
# Difference Metrics (p=prediction, t=truth)
# -------------------------------------------
def square_diff(p, t, plot=False):

    err = (p - t)**2

    if plot:
        plt.figure(2)
        pd.DataFrame(err).plot()
        plt.show()

    print('avg square diff: ', np.sum(err)/err.size)

    return np.sum(err)/err.size


def cosine_diff(p, t, axis=0, plot=False):

    s_up = np.sum(p*t, axis=axis)
    s1 = np.sqrt(np.sum(p**2, axis=axis))
    s2 = np.sqrt(np.sum(t**2, axis=axis))
    s = s_up / (s1 * s2)

    if plot:
        plt.figure(3)
        pd.DataFrame(1-s).plot()
        plt.show()

    print('avg cosine diff: ', np.mean(1-s))

    return np.mean(1 - s)


def classification_error(p, t):
    p_max = np.argmax(p, axis=1)
    t_max = np.argmax(t, axis=1)
    n = p.shape[0]
    ac = np.sum(p_max == t_max) / n

    print('trend accuracy', ac)

    return ac


def cross_entropy(p, t, eps=1e-9):
    p = np.clip(p, eps, 1. - eps)
    n = p.shape[0]
    ce = -np.sum(np.sum(t*np.log(p+eps))) / n

    print('cross_entropy', ce)

    return ce


def cross_corr(p, t, col=4):
    corr_list = []
    if col > 1:
        for i in range(col):
            sc = np.corrcoef(p[:, i], t[:, i])
            corr_list.append(sc[0, 1])
        corr = np.mean(corr_list)
    else:
        corr = np.corrcoef(p, t)[0, 1]

    print('crosscorr: ', corr)

    return corr


def spearman_corr(p, t, axis=0, col=4):

    s_corr = spearmanr(p, t, axis=axis)[0]
    s_tot = np.array([s_corr[i, i + col] for i in range(int(s_corr.shape[0]/2))])
    s_mean = np.mean(abs(s_tot))

    print('avg spearman correlation: ', s_mean)

    return s_mean

# ------------------------
# calculations for tuning
# ------------------------
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def tune_bandwidth(bw, qse, ref, sig):
    qse.n_support = int(bw[0]) # bw
    qse.delay = (bw - 1) / 2
    res_bw = qse.run(sig)
    qse.plot(sig, res_bw)
    result_bw = res_bw[:, 2 * qse.coeff_nr + qse.prim_nr:2 * qse.coeff_nr + 2 * qse.prim_nr + 1]
    return cosine_diff(ref, result_bw)



