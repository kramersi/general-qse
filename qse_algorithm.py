"""
Generalized Qualitative State Estimation (GQSE)

This Script is based on the QSE algorithm, which Kris Villez implemented in Matlab. This method based on the earlier
developed Qualitative Path estimation is further generalized by allowing all 39 different combinations of signs between
signal, first and second derivative and implements possibility
to choose from different kernels. Furthermore it is capable of handling different polynom order bigger than 1. This
version use is two times faster, because not calculated over a loop, but directly applying an array of moving windows.
In this case nans are not handled but signal should be interpolated first.

The standard method is well described in the following two papers:

    - Villez, K., & Rengaswamy, R. (2013, July). A generative approach to qualitative trend analysis for batch process
      fault diagnosis. In Control Conference (ECC), 2013 European (pp. 1958-1963). IEEE.
    - Thürlimann, C. M., Dürrenmatt, D. J., & Villez, K. (2015). Evaluation of qualitative trend analysis as a tool for
      automation. In Computer Aided Chemical Engineering (Vol. 37, pp. 2531-2536). Elsevier.


Copyright (C) 2018 Kris Villez, Simon Kramer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.


"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pylab as plt
import time


__all__ = ['order_validation', 'kernel_validation', 'non_negative_validation', 'extract_primitives',
           'generate_trans_matrix', 'GeneralQSE']

# ToDo: indicate that now not online but offline
# ToDo: make code also workable for Batch processing


def rolling_window(a, window):
    """ makes an array with all moving window entries at each time step using stride trick"""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)


def order_validation(order):
    """
    Validate the order. If it's under 2 then raise a ValueError

    """
    if order < 2:
        raise ValueError("An order lower than two is not allowed.")
    else:
        return order


def kernel_validation(kernel):
    """
    Validate if kernel type is one of the suggested kernel types and raise ValueEror if it's not so.

    """
    if kernel not in ['tricube', 'gaussian', 'epan']:
        raise ValueError("The kernel must be of type tricube, gaussian or epanechnikov.")
    else:
        return kernel


def non_negative_validation(value):
    """
    Validate if value is negative and raise Validation error

    """
    if value <= 0:
        raise ValueError("The Value must not be negative.")
    else:
        return value


def extract_primitives(transitions):
    """
    Extract all the primitives out of the possible transititions, which are defined by the user and make an list
    of unique primitives.

    Args:
        transitions (list): indicated start, end primitive and their transition probability

    Returns:
         list of all primitives

    """
    primitives = set()

    for transition in transitions:
        primitives.add(transition[0])  # collect start primitive
        primitives.add(transition[1])  # collect end primitive

    return sorted(list(primitives))  # transform set of unique primitives to list to be able to access index


def generate_trans_matrix(transitions, primitives):
    """
    Generate the transition matrix out of the list of transitions defined by start primitive, end primitve and
    transition probability factor.

    Args:
        transitions: list of all possible transitions between primitives
        primitives: list of all occuring primitives in the transtions

    Returns:
         transition_matrix with entries

    """
    prim_nr = len(primitives)
    transition_mat = np.zeros((prim_nr, prim_nr))  # initialize transition matrix (column=start, row=end)

    for tr in transitions:
        start_index = primitives.index(tr[0])  # get the index of start primitive
        end_index = primitives.index(tr[1])  # get the index of end primitive
        transition_mat[start_index, end_index] = tr[2]  # put the transition probability in right matrix place

    return transition_mat


class GeneralQSE(object):
    """ This class represent an generalisation of the Qualitative State Estimation method proposed by Kris Villez.

    It consists of the two steps Kernel Regression and Hidden Markov Chain (HMM). The kernel regression consists of all
    parameter and method to make a local polynom regression with different kernel types (tricube, gaussian, epanichikov)
    and different polynom orders (at least 2). The HMM is defined over a transtion list with defines probability of each
    transition between primitives.

    Args:
        sigma_eps (float): pre-supposed measurement error variance, its now not presupposed but calculated with
        standard deviation of filtered signal and raw signal.
        order (int): order of regression model, must be higher than 2 (2: linear, 3: quadratic, etc)
        delta (float): bandwidth parameter for the probability of zero-valued derivative. It indicates at which
            Percent of the maximum signal value, 1st and 2nd derivative the Probability to be in zero state is 68%.
            better name could be zero_threshold
        n_support (int): initial length of support window (often called bandwidth in kernel regression)
        bw_estimation (bool): indicates if bandwith should be estimated with cross validation or is user specific
    """
    # define all the transitions each sublist has start primitive, end primitive and transition probability in it.
    allowed_trans = [['Q', 'U+', 0.5], ['U+', 'L+', 1], ['L+', 'U+', 0.5], ['L+', 'Q0+', 1]]

    # all primitives are defined from all combination (4**3=64) only 39 make sense
    all_primitives = {
        # list entries:  [sign of signal, sign of first_derivative, sign of second_derivative]
        # positive values
        'F+': ['pos',     'zero',  'zero'],    # 'flat positive'
        'B+': ['pos',     'pos',   'pos'],     # 'boost positive'],
        'G+': ['pos',     'pos',   'zero'],    # 'equal growth positive'],
        'C+': ['pos',     'pos',   'neg'],     # 'concave deceleration positive'],
        'A+': ['pos',     'neg',   'pos'],     # 'antitonic acceleration positive'],
        'E+': ['pos',     'neg',   'zero'],    # 'equal decrease positive'],
        'D+': ['pos',     'neg',   'neg'],     # 'deceleration positive'],

        # negative values
        'F-': ['neg',     'zero',  'zero'],    # 'flat negative'],
        'B-': ['neg',     'pos',   'pos'],     # 'boost negative'],
        'G-': ['neg',     'pos',   'zero'],    # 'equal growth negative'],
        'C-': ['neg',     'pos',   'neg'],     # 'concave deceleration negative'],
        'A-': ['neg',     'neg',   'pos'],     # 'antitonic acceleration negative'],
        'E-': ['neg',     'neg',   'zero'],    # 'equal decrease negative'],
        'D-': ['neg',     'neg',   'neg'],     # 'deceleration negative'],

        # measured value ignored
        'F':  ['ignore',  'zero',  'zero'],    # 'flat'],
        'B':  ['ignore',  'pos',   'pos'],     # 'boost'],
        'G':  ['ignore',  'pos',   'zero'],    # 'equal'],
        'C':  ['ignore',  'pos',   'neg'],     # 'concave deceleration'],
        'A':  ['ignore',  'neg',   'pos'],     # 'antitonic acceleration'],
        'E':  ['ignore',  'neg',   'zero'],    # 'equal decrease'],
        'D':  ['ignore',  'neg',   'neg'],     # 'deceleration'],

        # just sign of signal all derivatives are ignored
        'Q0': ['zero',   'ignore', 'ignore'],  # 'zero signal'],
        'Q+': ['pos',    'ignore', 'ignore'],  # 'positive signal'],
        'Q-': ['neg',    'ignore', 'ignore'],  # 'negative signal'],

        # states where first derivative is ignored
        'P+': ['pos',    'ignore', 'pos'],    # 'positive positive'],
        'N+': ['pos',    'ignore', 'neg'],    # 'negative positive'],
        'O+': ['pos',    'ignore', 'zero'],   # 'line positive'],
        'P-': ['neg',    'ignore', 'pos'],    # 'positive negative'],
        'N-': ['neg',    'ignore', 'neg'],    # 'negative negative'],
        'O-': ['neg',    'ignore', 'zero'],   # 'line negative'],
        'P':  ['ignore', 'ignore', 'pos'],    # 'postive'],
        'N':  ['ignore', 'ignore', 'neg'],    # 'negative'],
        'O':  ['ignore', 'ignore', 'zero'],   # 'line'],

        # states where curvature (2nd derivative) is ignored
        'U+': ['pos',    'pos',    'ignore'],  # 'upper positive'],
        'L+': ['pos',    'neg',    'ignore'],  # 'lower positive'],
        'U-': ['neg',    'pos',    'ignore'],  # 'upper negative'],
        'L-': ['neg',    'neg',    'ignore'],  # 'lower negative'],
        'U':  ['ignore', 'pos',    'ignore'],  # 'upper'],
        'L':  ['ignore', 'neg',    'ignore'],  # 'lower']
    }

    # Precision of computer calculations, needed to prevent zero division
    precision = float(1e-16)

    def __init__(self, kernel='tricube', order=3, delta=0.05, transitions=allowed_trans, n_support=None, bw_estimation=False):
        # initialisation of kernel regression properties
        self.bw_estimation = bw_estimation
        self.n_support = n_support
        self.sigma_eps = 1
        self.order = order_validation(order)
        self.kernel = kernel_validation(kernel)
        self.delta = non_negative_validation(delta)
        self.coeff_nr = min(order, 3)
        if n_support is not None:
            self.delay = (n_support - 1)/2
        else:
            self.delay = None

        self.w = None
        self.regr_basis = None

        # initialization of hidden markov chain properties
        self.transitions = transitions
        primitives = extract_primitives(transitions)
        self.primitives = primitives
        self.prim_nr = len(primitives)
        self.trans_matrix = generate_trans_matrix(transitions, primitives)

    def set_bandwidth(self, bw):
        # check, because in fmin bw is passed as a ndarray and not an integer
        if isinstance(bw, np.ndarray):
            self.n_support = int(bw[0])
        else:
            self.n_support = bw

        self.delay = (bw - 1) / 2

    def get_projection_matrix(self):
        """
        calculates the projection matrix for kernel regression, which is dependend only on bandwith of kernel

        Args:
            n_support:

        Returns:
            projection matrix

        """
        n_width = self.n_support + 2  # kernel width (account for zeros at edge)
        half_width = (n_width - 1) / 2

        # regression input vector (relative time within support window)
        t_support = (np.arange(1, self.n_support + 1) - half_width) / half_width

        # regression basis functions:
        taylor_coeff = [n_width ** n / np.math.factorial(n) for n in range(self.order)]
        regr_basis = np.power.outer(t_support, np.arange(self.order)) * np.array(taylor_coeff)

        # regression weights
        if self.kernel == 'tricube':
            w = (1 - np.abs(t_support) ** 3) ** 3  # weight vector
        elif self.kernel == 'gaussian':
            w = (1. / np.sqrt(2 * np.pi)) * np.exp(-t_support ** 2 / (n_width ** 2 * 2.))
        elif self.kernel == 'epan':
            w = 0.75 * (1 - t_support ** 2)

        self.w = w
        self.regr_basis = regr_basis

        # steady state hat matrix (a.k.a. projection matrix):
        proj_matrix = np.linalg.inv((regr_basis.T * w).dot(regr_basis)).dot(regr_basis.T) * w

        return proj_matrix

    def coefficient_uncertainty(self, proj_matrix, y_stacks):
        """
        Get uncertainity of each polynom coefficient by taking diagonal entries of covariance matrix

        Arguments:
            dim (int): dimension of filtered signal
            sigma_eps (float): variance of error of residuals (measurment error)
            proj_matrix (ndarray): projection matrix

        Retruns:
            standard deviation of each coefficient in every time step

        """
        sig_n = y_stacks.shape[0]
        influence_matrix = (self.regr_basis.dot(proj_matrix))
        sigmas = np.full((sig_n, self.coeff_nr), np.nan)

        for i in range(sig_n):
            residuals = y_stacks.T[:, i] - np.dot(influence_matrix, y_stacks.T[:, i])
            sigma_eps = max(np.std(residuals), self.precision)
            sigma_b = (proj_matrix * sigma_eps).dot(proj_matrix.T)
            stdev_b = np.diag(sigma_b) ** (1 / 2)  # point-wise standard deviations out of diagonal covariance matrix
            stdev_b = stdev_b[0:self.coeff_nr]

            sigmas[i, :] = stdev_b

        return sigmas

    def initialize_bandwidth(self, signal):
        """
        This method calculates the initial bandwidth out of the data by a simple guess

        Args:
            signal (ndarray): data which used for kernel regression

        Returns:
            an initial bandwidth

        """
        stdev_data = np.std(signal)
        mean_data = np.mean(signal)
        # init_bw = 1.06 * stdev_data/mean_data * len(signal) ** (- 1. / (4 + self.order)) * len(signal)
        # print('ini_bw', init_bw)
        min_bw = 20
        init_bw = max(len(signal) * 0.05, min_bw) * stdev_data/mean_data
        print('initial bandwidth: ', init_bw)
        self.set_bandwidth(init_bw)

        return int(init_bw)

    def initialize_states(self):
        """
        Initialize the primitive probability states of the Hidden Markov Chain

        Returns:
             list of initial probability of states
        """
        states = [1.0 for _ in self.primitives]

        return np.array(states / np.sum(states))

    def predict_features(self, signal, proj_matrix):
        """
        Make polynomial kernel regression by creating array of all moving windows, calculating projection matrix and
        calculating.
        # Make signal longer at beginnig and end by the half of the delay, that filter exactly starts at first entry
        # and not half of delay later. It does so by taking the nearest value and multiplies it.
        Args:
            signal: signal to be smoothed
            n_support: bandwidth of kernel_regression

        Return:
            features=polynom-coefficient

        """
        # Extend signal at beginnig and end by the half of the moving window width, that filter exactly starts at first
        # entry and not a delay later. It does so by taking the nearest value and multiplies it.
        # If n_support is even number, add one value to back, that length correspond to each other.
        method = 'nearest'
        if method == 'nearest':
            first = np.repeat(signal[0], self.delay)
            if self.n_support % 2 == 1:
                last = np.repeat(signal[-1], self.delay)
            else:
                last = np.repeat(signal[-1], self.delay+1)
        elif method == 'same':
            first = signal[0:int(self.delay)]
            if self.n_support % 2 == 1:
                last = signal[-int(self.delay):-1]
            else:
                last = signal[-int(self.delay)-1:-1]

        extended_signal = np.concatenate((first, signal, last))

        # n*m array of moving windows, n=signal_length, m=n_support
        y_stacks = rolling_window(extended_signal, self.n_support)

        # only if all values are not nan, but this is not the case because signal interpolated at beginning
        features = proj_matrix.dot(y_stacks.T).T

        return features[:, 0:self.coeff_nr], y_stacks

    def infer_probabilities(self, features, stdev):
        """ calculate probabilities of selected primitives out of polynom features and their standard deviation
        assuming that the features are independent and normal distributed with mean = coefficient it self and stdev.

        Args:
            features (ndarray): polynom coefficients
            stdev (ndarray): standard deviation of the polynom coefficients

        """
        b = features / stdev  # normalise features by their standard deviation, necessary to use norm.cdf

        # get the max of each feature over time and take a procentage of it
        deltas = np.nanmax(b, axis=0) * self.delta

        # Convert regression coefficients to probabilities for qualitative state
        prob_p = norm.logcdf(b - deltas)
        prob_n = norm.logcdf(-b - deltas)
        prob_pn = norm.logcdf(deltas - b)
        prob_0 = prob_pn + np.log1p(-np.exp(prob_n-prob_pn))

        params = {'pos': prob_p, 'neg': prob_n, 'zero': prob_0, 'ignore': np.zeros(b.shape)}

        py = np.zeros((b.shape[0], self.prim_nr))

        # collect for each primitive the right probability and multiplies them together
        for i, prim in enumerate(self.primitives):  # iterate over primitives
            # iterate over signal, 1st and 2nd derivatives, this is equal to iterate over params of polynom.
            prob = [params[sign][:, j] for j, sign in enumerate(self.all_primitives[prim]) if j < self.coeff_nr]
            py[:, i] = np.sum(prob, axis=0)

        return py

    def estimate_states(self, primitive_prob, states):
        """
        Estimates the new states out of transition matrix, primitive probabilities and old state.

        Args:
            states (ndarray): old probability state
            primitive_prob (ndarray): probability of primitives

        Returns:
             states (ndarray): new probabilty state

        """
        if not np.any(np.isnan(primitive_prob)):
            trans_states = np.dot(self.trans_matrix, states)
            states = primitive_prob + np.log(trans_states)
            prob_n = states - np.max(states)  # introduce scaling with max value for normalisation afterwards
            states = np.exp(prob_n) / sum(np.exp(prob_n))
        return states

    def pmse_gcv(self, bw, signal):
        """
        calculating the predicted regression error if leaving one out cross validation is done. But instead of
        iterating n times over the signal length by leaving one away, the error is estimated whith the generalised
        cross validation approximation. Therefore just one iteration is needed per bandwidth estimation.

        Args:
            y: signal which should be smoothed
            yhat: filtered signal by local polynom regression

        Returns:
            estimated error introduced by gcv

        """
        self.set_bandwidth(bw)
        print('try bandwidth ... ', bw)

        # calculate projection matrix and update regr_basis and weights changed by new n_support
        proj_matrix = self.get_projection_matrix()

        # make kernel regression by applying moving window of signal to projection matrix
        all_features, y_stacks = self.predict_features(signal, proj_matrix)

        # # extend signal
        # first = np.repeat(signal[0], self.delay)
        # if self.n_support % 2 == 1:
        #     last = np.repeat(signal[-1], self.delay)
        # else:
        #     last = np.repeat(signal[-1], self.delay + 1)
        #
        # ext_signal = np.concatenate((first, signal, last))


        # make signals to equal length and shift it to calculate residuals
        # raw = np.delete(signal, np.arange(self.n_support - 1))
        # signal = pd.Series(signal)
        # filtered = all_features[:, 0]
        # residuals = signal - filtered  # .shift(-int(self.delay))
        # #cov_mat = np.cov((signal, filtered), rowvar=0)
        # print('resSumBef', np.nansum(residuals))
        # plt.figure(4)
        # plt.subplot(2, 1, 1)
        # pd.Series(signal).plot()
        # filtered.plot()
        # plt.subplot(2, 1, 2)
        # residuals.plot(kind='hist', bins=40)
        # plt.show()

        # calculating influence matrix, which is needed for calculating gcv afterwards
        # basis = self.regr_basis
        # w = self.w
        # # Simplify because in linear projection trace of infl matrix is always degegree of freedom (order)
        # influence_matrix = (basis.dot(proj_matrix))
        # trace = sum(np.diag(influence_matrix))
        # sig_n = len(signal)
        # n = self.n_support
        #trace = self.order
        #neg_inf = np.eye(n) - influence_matrix
        #n_inv = 1/n

        # # gcv score with general cross validation
        # n = signal.size
        # gccv = np.full(n, np.nan)
        # influence_matrix = (self.regr_basis.dot(proj_matrix))
        # for i in range(self.n_support):
        #     s = influence_matrix
        #     y = np.dot(influence_matrix, y_stacks.T)[:,i]
        #     x = y_stacks.T[:, i]
        #     sig = np.std(x-y)
        #     c = np.cov((x, y), rowvar=0)/sig
        #     gccv[i] = 1/n * np.nansum((x-y)**2) / (1 - sum(np.diag(2*s.dot(c) - np.dot(s.dot(c),s.T)))/n)**2
        # print('gccv', gccv)

        # gcv without cross correlation
        sig_n = signal.size
        n = self.n_support
        gcv = np.full(sig_n, np.nan)
        influence_matrix = (self.regr_basis.dot(proj_matrix))
        trace = self.order
        for i in range(sig_n):
            residuals = y_stacks.T[:, i] - np.dot(influence_matrix, y_stacks.T[:, i])
            gcv[i] = 1 / n * np.nansum(residuals ** 2) / (1 - trace / n) ** 2  # Generated cross validation
        # # gccv score with genarela correlated cross validation C is calc with outer(res, res)
        # sig_n = signal.size
        # n = self.n_support
        # gccv = np.full(sig_n, np.nan)
        # influence_matrix = (self.regr_basis.dot(proj_matrix))
        # for i in range(sig_n):
        #     s = influence_matrix
        #     y = np.dot(influence_matrix, y_stacks.T[:,i])
        #     x = y_stacks.T[:, i]
        #     res = x - y
        #     sig = np.std(res)
        #     if sig > 0:
        #         c = np.outer(res, res)/sig
        #         gccv[i] = 1 / n * np.nansum(res ** 2) / (
        #                     1 - sum(np.diag(2 * s.dot(c) - np.dot(s.dot(c), s.T))) / n) ** 2
        #     else:
        #         gccv[i] = 0
        #
        # print('gccv', gccv)
        # print('score', np.nanmean(gccv))

        # # gcv score with overall smooth matrix
        # sig_n = len(signal)
        # sm = np.full((sig_n, sig_n+self.n_support-1), 0.0)
        # for i in range(sig_n):
        #     ni = 0
        #     if self.n_support % 2 == 1:
        #         ni = 0
        #     sm[i, i:self.n_support+i+ni] = proj_matrix[0, :]
        # # sig = np.std(residuals)
        # # c = np.outer(residuals, residuals)/(sig+0.00001)
        # # smc = sm.T.dot(c)
        # #gccvg = 1 / sig_n * np.nansum((residuals) ** 2) / (1 - sum(np.diag(2 * smc) - np.diag(np.dot(smc, sm))) / sig_n) ** 2
        # #print(sm)
        # #print(gccvg)
        # trace = sum(np.diag(sm, 1))
        # # print('sizeofREs', res.shape)
        # gcv_score = (1/sig_n) * np.nansum((residuals/(1 - trace/sig_n))**2)
        # # print('score: ', gcv_score, gcv_sc)

        return np.max(gcv)  # np.mean(gcv_sc)

    def sigma_eps_estimation(self, raw, filtered, show_results=False):
        """ Extract the variance of the residuals

        Attributes:
            raw (array): raw signal vector
            filtered (array): filtered signal with polynomial kernel smoothing
            show_results (bool): inidicates if plot of residuals should be plotted

        Returns:
            standard deviation of residuals

        """
        # raw = np.delete(raw, np.arange(self.n_support-1))
        residuals = pd.Series(raw - filtered)  # .shift(-int(self.delay))
        # print(np.cov((raw.values, filtered.values), rowvar=0))
        if show_results is True:
            plt.figure(3)
            plt.subplot(3, 1, 1)
            residuals.plot()
            plt.subplot(3, 1, 2)
            residuals.plot(kind='hist', bins=40)
            plt.subplot(3, 1, 3)
            raw.plot()
            filtered.plot()
            plt.show()

        return residuals.std()

    def run(self, signal):
        """
        Run the QSE algorithme, by iterating over each data point in the signal

        Args:
            signal (ndarray): input signal which should be processed

        Returns:
            memory: array with all states, primitive probabilities etc.

        """

        if self.n_support is None:
            bw_init = self.initialize_bandwidth(signal)
        else:
            bw_init = self.n_support

        if self.bw_estimation is True:
            # methods: TNC, SLSQP, Nelder-Mead, COBYLA, L-BFGS-B
            #cons = ({'type': 'eq', 'fun': lambda x: max(x-int(x))},)
            # minimum = minimize(self.pmse_gcv, bw_init, method='Nelder-Mead', args=(signal,))
            # minimum = minimize(self.pmse_gcv, bw_init, method='SLSQP', args=(signal,), bounds=((3, None),), constraints=cons)  # , options={'xatol': 1.0})
            # minimum = minimize(self.pmse_gcv, bw_init, method='COBYLA', args=(signal,))
            # minimum = minimize(self.pmse_gcv, bw_init, method='L-BFGS-B', args=(signal,), bounds=((3, None),))
            # minimum = fmin(self.pmse_gcv, bw_init, args=(signal, ), xtol=1)
            # print(minimum)
            hs = np.linspace(bw_init/20, bw_init*2, dtype='int16')

            maxgcv = np.full(len(hs), np.nan)
            for j, n in enumerate(hs):
                maxgcv[j] = self.pmse_gcv(n, signal)

            # gcv_df = pd.DataFrame({'bandwidth': np.array(hs), 'max_gcv': maxgcv})
            # gcv_df.to_csv('gcv_results/gcv.csv')

            plt.figure(3)
            plt.plot(np.array(hs), maxgcv)
            plt.xlabel('Bandwidth [#]')
            plt.ylabel('GCV-Score')
            plt.show()

            self.set_bandwidth(hs[np.argmin(maxgcv)])
            print('best bandwidth found: ', self.n_support)

        proj_matrix = self.get_projection_matrix()
        all_features, y_stacks = self.predict_features(signal, proj_matrix)

        # # extract epsilon by calculating residuals between raw signal and filtered signal
        # sigma_eps= self.sigma_eps_estimation(signal, all_features[:, 0])

        # redo kernel loop now with new estimated sigma calculated out of residuals
        all_stdev = self.coefficient_uncertainty(proj_matrix, y_stacks)

        # calculate the probabilieties out of features and stdev
        all_primitive_prob = self.infer_probabilities(all_features, all_stdev)

        # make hidden markov model by iteration over each timestep
        states = self.initialize_states()  # initailization of states, all equal probability
        all_states = np.full((signal.size, self.prim_nr), np.nan)
        for i, prim_prob in enumerate(all_primitive_prob):
            states = self.estimate_states(prim_prob, states)
            all_states[i, :] = states

        memory = np.hstack((all_features, all_stdev, np.exp(all_primitive_prob), all_states))

        return memory

    def plot(self, signal, memory):
        """
        Plot the signal and the smoothed signal, the primitive probabilities and the primitive states.

        Args:
            memory (ndarray): values of the result of the QSE (nr, signal, features, primitive prob., primitive states)
            signal (ndarray): raw signal
        """
        offset = 2 * self.coeff_nr
        state_range = np.arange(offset + self.prim_nr, offset + 2 * self.prim_nr)
        prim_range = np.arange(offset, offset + self.prim_nr)
        nr = np.arange(memory.shape[0])
        plt.figure(1)

        # plot signal and filtered signal
        p2 = plt.subplot(3, 1, 1)
        # plt.plot(nr, signal[self.n_support-2:-1], 'k-')
        plt.plot(nr, signal, 'k-')
        # plt.plot(nr - self.delay, memory[:, 0], 'b-')
        plt.plot(nr, memory[:, 0], 'b-')
        plt.xlabel('Sample index [-]')
        plt.ylabel('Signal value [-]')
        plt.legend(('raw signal', 'filtered signal'))

        # plot primitive probabilities
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        plt.subplot(3, 1, 2, sharex=p2)
        # plt.plot(nr - self.delay, memory[:, prim_range], '-')
        for i, r in enumerate(prim_range):
            plt.plot(nr, memory[:, r], '-', color=colors[i])
        plt.xlabel('Sample index [-]')
        plt.ylabel('Primitive probabilities [-]')
        plt.legend(tuple(self.primitives))

        # plot states probabilities
        plt.subplot(3, 1, 3, sharex=p2)
        # plt.plot(nr - self.delay, memory[:, state_range], '-')
        for i, r in enumerate(state_range):
            plt.plot(nr, memory[:, r], '-', color=colors[i])
        plt.xlabel('Sample index [-]')
        plt.ylabel('HMM state probabilities [-]')
        plt.legend(tuple(self.primitives))

        # plot feature derivatives and their confidence interval
        plt.figure(2)
        for i in range(self.coeff_nr):
            plt.subplot(self.coeff_nr, 1, i+1)
            plt.plot(nr, memory[:, i], '-')
            plt.plot(nr, memory[:, i] + 2 * memory[:, self.coeff_nr+i], '--')
            plt.plot(nr, memory[:, i] - 2 * memory[:, self.coeff_nr+i], '--')

            plt.xlabel('Sample index [-]')
            plt.ylabel('derivative: ' + str(i))

        plt.show(block=True)


if __name__ == '__main__':
    file = 'cam1_intra_0_0.2_0.4__ly4ftr16w2__cam1_0_0.2_0.4.csv'

    # Part I. load data from csv
    df = pd.read_csv('data/'+file, sep=',', dtype={'sensor_value': np.float64})
    df = df.interpolate()

    #y = df['flood_index'].values
    y = df['sensor_value'].values

    # Part II. Algorithm setup and run
    # A. Setup and Initialization with tunning parameters
    epsi = 0.000001
    trans1 = [['Q0', 'Q0', 0.50], ['Q0', 'L+', epsi], ['Q0', 'U+', 0.50], ['Q0', 'F+', epsi],
              ['L+', 'Q0', 0.25], ['L+', 'L+', 0.25], ['L+', 'U+', 0.25], ['L+', 'F+', 0.25],
              ['U+', 'Q0', epsi], ['U+', 'L+', 0.33], ['U+', 'U+', 0.33], ['U+', 'F+', 0.33],
              ['F+', 'Q0', epsi], ['F+', 'L+', 0.33], ['F+', 'U+', 0.33], ['F+', 'F+', 0.33]]

    trans2 = [['Q0', 'Q0', 0.50], ['Q0', 'L+', epsi], ['Q0', 'U+', 0.50], ['Q0', 'F+', epsi],
              ['L+', 'Q0', 0.33], ['L+', 'L+', 0.33], ['L+', 'U+', epsi], ['L+', 'F+', 0.33],
              ['U+', 'Q0', epsi], ['U+', 'L+', epsi], ['U+', 'U+', 0.50], ['U+', 'F+', 0.50],
              ['F+', 'Q0', epsi], ['F+', 'L+', 0.33], ['F+', 'U+', 0.33], ['F+', 'F+', 0.33]]

    qse = GeneralQSE(kernel='tricube', order=3, delta=0.05, transitions=trans2, bw_estimation=True, n_support=300)

    # B. Run algorithms
    t = time.process_time()
    result = qse.run(y)
    elapsed_time = time.process_time() - t
    print('time requirements: ', elapsed_time)
    print('finish')

    # Part III: Display
    qse.plot(y, result)
