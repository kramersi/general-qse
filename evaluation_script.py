import numpy as np
import pandas as pd
from qse_algorithm import GeneralQSE
from diff_metrics import square_diff, cosine_diff
from os import listdir


def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


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

    run_configs = [dict(col='sensor_value', bw=200, trans=trans2, bw_est=True),
                  dict(col='flood_index', bw=200, trans=trans2, bw_est=False)]

    # run_configs = [dict(col='sensor_value', bw=None, trans=trans2, bw_est=True)]
    result = np.full((df.shape[0], 4, len(run_configs)), np.nan)

    for i, rc in enumerate(run_configs):
        qse = GeneralQSE(kernel='tricube', order=3, delta=0.05, transitions=rc['trans'], bw_estimation=rc['bw_est'], n_support=rc['bw'])
        signal = df[rc['col']].values
        res = qse.run(signal)
        result[:, :, i] = res[:, 2 * qse.coeff_nr + qse.prim_nr:2 * qse.coeff_nr + 2 * qse.prim_nr + 1]
        qse.plot(signal, res)

    diff_all['sqe'].append(square_diff(result[:, :, 0], result[:, :, 1]))
    diff_all['cos'].append(cosine_diff(result[:, :, 0], result[:, :, 1]))