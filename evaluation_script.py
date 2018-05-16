import numpy as np
import pandas as pd
from qse_algorithm import GeneralQSE
from diff_metrics import square_diff, cosine_diff, spearman_corr, cross_corr
from scipy.optimize import minimize
from scipy.stats import spearmanr
from os import listdir
from os.path import isfile
import matplotlib.pylab as plt


def information_thres(x, y):
    # ToDo: include a threshold for the spearman correlation if to low, don't use refernce and sofi pair for tuning parameter.
    corr = cross_corr(x, y, col=1)
    return corr > 0.5


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


diff_all = dict(file=[], cosine_diff_0=[], cosine_diff_1=[], sqe_diff=[], correlation_trend=[], correlation_signal=[])
path = '/Users/simonkramer/PycharmProjects/MasterThesis/general-qse/data'
path_store = '/Users/simonkramer/PycharmProjects/MasterThesis/general-qse/results180516'

# read in files
if 0:
    for f in find_csv_filenames(path):
        file = path + '/' + f
        print('evaluate csv-file: ', f)
        df = pd.read_csv(file, sep=',', dtype={'sensor_value': np.float64})
        df = df.interpolate()

        print(df.corr(method='pearson'))
        epsi = 0.000001

        trans2 = [['Q0', 'Q0', 0.50], ['Q0', 'L+', epsi], ['Q0', 'U+', 0.50], ['Q0', 'F+', epsi],
                  ['L+', 'Q0', 0.33], ['L+', 'L+', 0.33], ['L+', 'U+', epsi], ['L+', 'F+', 0.33],
                  ['U+', 'Q0', epsi], ['U+', 'L+', epsi], ['U+', 'U+', 0.50], ['U+', 'F+', 0.50],
                  ['F+', 'Q0', epsi], ['F+', 'L+', 0.33], ['F+', 'U+', 0.33], ['F+', 'F+', 0.33]]

        run_configs = [dict(col='sensor_value', bw=300, trans=trans2, bw_est='ici', bw_tune=False),
                      dict(col='flood_index', bw=300, trans=trans2, bw_est='ici', bw_tune=False)]

        # run_configs = [dict(col='sensor_value', bw=None, trans=trans2, bw_est=True)]
        result = np.full((df.shape[0], 5, len(run_configs)), np.nan)

        for i, rc in enumerate(run_configs):
            new_path = 'results180516/' + f.split('.cs')[0] + '-' + rc['col']
            if not isfile(new_path + '.csv'):
                qse = GeneralQSE(kernel='tricube', order=3, delta=0.03, transitions=rc['trans'], bw_estimation=rc['bw_est'],
                                 n_support=rc['bw'])
                signal = df[rc['col']].values

                if rc['bw_tune']:
                    # scores = []
                    # bws = [5, 10, 15, 20, 40, 60, 100, 120, 160, 200, 250, 300, 400, 500, 600, 700, 800]
                    # for bw in bws:
                    #     scores.append(tune_bandwidth(bw, qse, result[:, :, 0], signal))
                    #     bw_best = bws[np.argmin(scores)]
                    # print('bw won', bw_best)
                    minimum = minimize(tune_bandwidth, np.array([rc['bw']]), method='Nelder-Mead', args=(qse, result[:, :, 0], signal))
                    qse.n_support = int(minimum['x'])  # bws[np.argmin(scores)]
                    qse.delay = float((int(minimum['x']) - 1) / 2)
                res = qse.run(signal)
                pd.DataFrame(res).to_csv(new_path + '.csv')
                result[:, :, i] = res[:, 2 * qse.coeff_nr + qse.prim_nr:2 * qse.coeff_nr + 2 * qse.prim_nr + 1]
                qse.plot(signal, res, path=new_path)

# calculate_difference
if 1:
    #result = np.full((df.shape[0], 5, len(run_configs)), np.nan)
    df_pair = dict(sensor_value=[], flood_index=[])
    files = find_csv_filenames(path_store)
    print(files)
    for i, f in enumerate(sorted(files)):
        if f.split('-')[1] == 'sensor_value.csv':
            df_pair['sensor_value'].append(f)
            diff_all['file'].append(f.split('-')[0])
        else:
            df_pair['flood_index'].append(f)

    for j, _ in enumerate(df_pair['sensor_value']):
        df_sens = pd.read_csv('results180516/' + df_pair['sensor_value'][j], sep=',')
        df_sofi = pd.read_csv('results180516/' + df_pair['flood_index'][j], sep=',')

        coeff_nr = 3
        prim_nr = 4

        sig_sens = df_sens.values[:, 1]
        sig_sofi = df_sofi.values[:, 1]

        feat_sens = df_sens.values[:, 1 + 2 * coeff_nr + prim_nr:2 * coeff_nr + 2 * prim_nr + 1]
        feat_sofi = df_sofi.values[:, 1 + 2 * coeff_nr + prim_nr:2 * coeff_nr + 2 * prim_nr + 1]

        # plt.figure(5)
        # plt.subplot(2, 1, 1)
        # plt.plot(feat_sens)
        # plt.subplot(2, 1, 2)
        # plt.plot(feat_sofi)
        # plt.show()
        sig_corr = cross_corr(sig_sens, sig_sofi, col=1)

        diff_all['sqe_diff'].append(square_diff(feat_sens, feat_sofi))
        diff_all['cosine_diff_0'].append(cosine_diff(feat_sens, feat_sofi, axis=0))
        diff_all['cosine_diff_1'].append(cosine_diff(feat_sens, feat_sofi, axis=1))
        diff_all['correlation_trend'].append(cross_corr(feat_sens, feat_sofi, col=prim_nr))
        diff_all['correlation_signal'].append(cross_corr(sig_sens, sig_sofi, col=1))

    pd.DataFrame(diff_all).to_csv(path_store + '/correlation_results.csv')
