import numpy as np
import pandas as pd
from qse_algorithm import GeneralQSE
import matplotlib.pylab as plt
from os import listdir


def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def square_error(x1, x2, plot=False):
    err = (x1 - x2)**2
    if plot:
        plt.figure(2)
        pd.DataFrame(err).plot()
        plt.show()
    print('avg difference: ', np.sum(err)/err.size)
    return np.sum(err)


def cosine_similiarity(x1, x2, plot=False):
    s_up = np.sum(x1*x2, axis=1)
    s1 = np.sqrt(np.sum(x1**2, axis=1))
    s2 = np.sqrt(np.sum(x2**2, axis=1))
    s = s_up / (s1 * s2)
    return np.sum(1 - s)


if __name__ == '__main__':
    diff_all=dict(cos=[], sqe=[])
    path = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/QualitativeTrendAnalyis/8_compute_index'
    for f in find_csv_filenames(path):
        file = path + '/' + f
        print('csv-file: ', f)
        df = pd.read_csv(file, sep=',', dtype={'sensor_value': np.float64})
        df = df.interpolate()

        lan = 1.0
        trans1 = [['Q0', 'Q0', 0.50 * lan], ['Q0', 'L+', 0.001 * lan], ['Q0', 'U+', 0.50 * lan], ['Q0', 'F+', 0.001 * lan],
                  ['L+', 'Q0', 0.25 * lan], ['L+', 'L+', 0.25 * lan], ['L+', 'U+', 0.25 * lan], ['L+', 'F+', 0.25 * lan],
                  ['U+', 'Q0', 0.25 * lan], ['U+', 'L+', 0.25 * lan], ['U+', 'U+', 0.25 * lan], ['U+', 'F+', 0.25 * lan],
                  ['F+', 'Q0', 0.01 * lan], ['F+', 'L+', 0.33 * lan], ['F+', 'U+', 0.33 * lan], ['F+', 'F+', 0.33 * lan]]
        trans2 = [['Q0', 'Q0', 0.50 * lan], ['Q0', 'L+', 0.01 * lan], ['Q0', 'U+', 0.50 * lan], ['Q0', 'F+', 0.01 * lan],
                  ['L+', 'Q0', 0.25 * lan], ['L+', 'L+', 0.25 * lan], ['L+', 'U+', 0.25 * lan], ['L+', 'F+', 0.25 * lan],
                  ['U+', 'Q0', 0.25 * lan], ['U+', 'L+', 0.25 * lan], ['U+', 'U+', 0.25 * lan], ['U+', 'F+', 0.25 * lan],
                  ['F+', 'Q0', 0.001 * lan], ['F+', 'L+', 0.33 * lan], ['F+', 'U+', 0.33 * lan], ['F+', 'F+', 0.33 * lan]]

        run_configs = [dict(col='flood_index', bw=200, trans=trans1),
                      dict(col='sensor_value', bw=200, trans=trans2)]
        result = np.full((df.shape[0], 4, len(run_configs)), np.nan)

        for i, rc in enumerate(run_configs):
            qse = GeneralQSE(kernel='tricube', order=3, delta=0.09, transitions=rc['trans'], bw_estimation=False, n_support=rc['bw'])
            signal = df[rc['col']].values
            res = qse.run(signal)
            result[:, :, i] = res[:, 2 * qse.coeff_nr + qse.prim_nr:2 * qse.coeff_nr + 2 * qse.prim_nr + 1]
            qse.plot(signal, res)

        diff_all['sqe'].append(square_error(result[:, :, 0], result[:, :, 1]))
        diff_all['cos'].append(cosine_similiarity(result[:, :, 0], result[:, :, 1]))

    print(diff_all)