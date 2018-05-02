import numpy as np
import pandas as pd
from qse_algorithm import GeneralQSE
from diff_metrics import square_diff, cosine_diff
from scipy.optimize import minimize
from os import listdir

def information_thres():
    # ToDo: include a threshold for the spearman correlation if to low, don't use refernce and sofi pair for tuning parameter.
    return 1

def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def tune_bandwidth(bw, qse, ref, sig):
    qse.n_support = int(bw[0]) # bw
    qse.delay = (bw - 1) / 2
    res_bw = qse.run(sig)
    qse.plot(sig, res_bw)
    result_bw = res_bw[:, 2 * qse.coeff_nr + qse.prim_nr:2 * qse.coeff_nr + 2 * qse.prim_nr + 1]
    return cosine_diff(ref, result_bw)


diff_all=dict(cos=[], sqe=[])
path = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/QualitativeTrendAnalyis/8_compute_index'

for f in find_csv_filenames(path):
    file = path + '/' + f
    print('evaluate csv-file: ', f)
    df = pd.read_csv(file, sep=',', dtype={'sensor_value': np.float64})
    df = df.interpolate()

    epsi = 0.000001

    trans2 = [['Q0', 'Q0', 0.50], ['Q0', 'L+', epsi], ['Q0', 'U+', 0.50], ['Q0', 'F+', epsi],
              ['L+', 'Q0', 0.33], ['L+', 'L+', 0.33], ['L+', 'U+', epsi], ['L+', 'F+', 0.33],
              ['U+', 'Q0', epsi], ['U+', 'L+', epsi], ['U+', 'U+', 0.50], ['U+', 'F+', 0.50],
              ['F+', 'Q0', epsi], ['F+', 'L+', 0.33], ['F+', 'U+', 0.33], ['F+', 'F+', 0.33]]

    run_configs = [dict(col='sensor_value', bw=80, trans=trans2, bw_est=False, bw_tune=False),
                  dict(col='flood_index', bw=200, trans=trans2, bw_est=False, bw_tune=True)]

    # run_configs = [dict(col='sensor_value', bw=None, trans=trans2, bw_est=True)]
    result = np.full((df.shape[0], 4, len(run_configs)), np.nan)

    for i, rc in enumerate(run_configs):
        qse = GeneralQSE(kernel='tricube', order=3, delta=0.05, transitions=rc['trans'], bw_estimation=rc['bw_est'],
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
        result[:, :, i] = res[:, 2 * qse.coeff_nr + qse.prim_nr:2 * qse.coeff_nr + 2 * qse.prim_nr + 1]
        qse.plot(signal, res)

    diff_all['sqe'].append(square_diff(result[:, :, 0], result[:, :, 1]))
    diff_all['cos'].append(cosine_diff(result[:, :, 0], result[:, :, 1]))