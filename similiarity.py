import numpy as np
import pandas as pd
from qse_algorithm import GeneralQSE
import matplotlib.pylab as plt

def similiarity(x1, x2):
    err = (x1 - x2)**2
    plt.figure(1)
    pd.DataFrame(err).plot()
    plt.show()
    return np.sum(err) / (err.size)

if __name__ == '__main__':
    df = pd.read_csv('data/cam1_intra_0_5_10__ly4ftr16__cam1_0_5_10.csv', sep=',', dtype={'sensor_value': np.float64})
    df = df.interpolate()

    lan = 1.0
    trans1 = [['Q0', 'Q0', 0.50 * lan], ['Q0', 'L+', 0.00 * lan], ['Q0', 'U+', 0.50 * lan], ['Q0', 'F+', 0.00 * lan],
              ['L+', 'Q0', 0.25 * lan], ['L+', 'L+', 0.25 * lan], ['L+', 'U+', 0.25 * lan], ['L+', 'F+', 0.25 * lan],
              ['U+', 'Q0', 0.25 * lan], ['U+', 'L+', 0.25 * lan], ['U+', 'U+', 0.25 * lan], ['U+', 'F+', 0.25 * lan],
              ['F+', 'Q0', 0.00 * lan], ['F+', 'L+', 0.33 * lan], ['F+', 'U+', 0.33 * lan], ['F+', 'F+', 0.33 * lan]]
    trans2 = [['Q0', 'Q0', 0.50 * lan], ['Q0', 'L+', 0.00 * lan], ['Q0', 'U+', 0.50 * lan], ['Q0', 'F+', 0.00 * lan],
              ['L+', 'Q0', 0.25 * lan], ['L+', 'L+', 0.25 * lan], ['L+', 'U+', 0.25 * lan], ['L+', 'F+', 0.25 * lan],
              ['U+', 'Q0', 0.25 * lan], ['U+', 'L+', 0.25 * lan], ['U+', 'U+', 0.25 * lan], ['U+', 'F+', 0.25 * lan],
              ['F+', 'Q0', 0.00 * lan], ['F+', 'L+', 0.33 * lan], ['F+', 'U+', 0.33 * lan], ['F+', 'F+', 0.33 * lan]]

    run_configs = [dict(col='flood_index', bw=85, trans=trans1),
                  dict(col='sensor_value', bw=30, trans=trans2)]
    result = np.full((df.shape[0], 4, len(run_configs)), np.nan)
    for i, rc in enumerate(run_configs):
        qse = GeneralQSE(kernel='tricube', order=3, delta=0.09, transitions=rc['trans'], bw_estimation=False, n_support=rc['bw'])
        signal = df[rc['col']].values
        res = qse.run(signal)
        print('coeff-prim', qse.coeff_nr, qse.prim_nr)
        result[:, :, i] = res[:, 2 * qse.coeff_nr + qse.prim_nr:2 * qse.coeff_nr + 2 * qse.prim_nr + 1]
        qse.plot(signal, res)
    print(similiarity(result[:,:,0], result[:,:,1]))

