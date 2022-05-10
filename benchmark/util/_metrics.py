import numpy as np
import pandas
import pandas as pd

from ._data_handling import BenchmarkPOD


def stability_score(pod: BenchmarkPOD):
    """
        Calculates a selection stability score for randomized selection methods.
        If r is the number of runs, n the number of features to select and d the number of distinct features selected
        across the runs, the score in range [0,1] is calculated as:

                        score = (n - (d - n)/(r -1)) / n

        Parameters
        ----------
        pod : BenchmarkPOD
            data container produced by benchmarking

        Returns
        -------
            Extends the passed BenchmarkPOD with the stability score according to the above formula

    """
    for n in pod.n_features:
        for method in pod.methods:
            for dataset in pod.datasets:
                data = pod.get_selection_data(method=method, n_features=n).to_numpy()
                d = np.unique(data).size
                r = pod.n_runs
                score = (n - (d - n)/(r - 1)) / n

                pod.register_stability(dataset=dataset, method=method, n_features=n, metric_name='stability_score', value=score)


def _intersection_expectation(n: int, s: int):
    """
        Calculates the expectation value of the size of the intersection of two samples of size s each drawn from the
        same pool of n unique entities without replacement

        Parameters
        ----------
        n : int
            number of elements drawn from
        s : int
            sample size

        Returns
        -------
        expectation value of the intersection size: float
    """
    logs = np.log(np.arange(1, n + 1))
    constant = 2 * np.sum(logs[:s]) + 2 * np.sum(logs[:n - s]) - np.sum(logs)

    m_fac = - logs[0]
    sm_fac = -2 * np.sum(logs[:s - 1])
    nssm_fac = - np.sum(logs[:n - 2 * s + 1])

    e = 0
    for i in range(s - 1):
        e += np.exp(m_fac + sm_fac + nssm_fac + logs[i] + constant)
        m_fac -= logs[i + 1]
        sm_fac += 2 * logs[s - i - 2]
        nssm_fac -= logs[n - 2*s + i + 1]

    # special case intersection of size s
    e += np.exp(logs[s - 1] + np.sum(logs[:s]) + np.sum(logs[:n - s]) - np.sum(logs))
    return e


def deng_stability_score(pod: BenchmarkPOD):
    """
            Calculates a selection stability score for randomized selection methods, according to Deng et al. [1]_.

            Parameters
            ----------
            pod : BenchmarkPOD
                data container produced by benchmarking

            Returns
            -------
                Extends the passed BenchmarkPOD with the stability scores calculated according to Deng et al. [1]_

            References
            ----------
            .. [1] Bai-Chuan Deng, Yong-Huan Yun, Pan Ma, Chen-Chen Li, Da-Bing Ren and Yi-Zeng Liang,
                   'A new method for wavelength interval selection that intelligently optimizes the locations, widths
                   and combination of intervals',
                   Analyst, 6, 1876-1885, 2015.

    """
    n_wavelengths = pod.get_meta_item("n_wavelengths")

    for method in pod.get_methods():
        supports = pod.get_item(method, "support")

        r = len(supports)
        pairwise_sim = np.empty(int((r**2 - r) / 2), dtype='float')

        sample_size = supports[0].shape[0]
        e = _intersection_expectation(n_wavelengths, sample_size)
        for i, (x, y) in enumerate(np.triu_indices(r)):
            if x != y:
                pairwise_sim[i] = (np.intersect1d(supports[x, y]) - e) / (np.sqrt(supports[x].shape[0] * supports[y].shape[0])
                                                                      - e)
        score = np.sum(pairwise_sim) * (2 / (r*(r - 1)))

        pod.register(method, deng_stability_score=score)


def mean_std_statistics(pod: BenchmarkPOD):

    """

        Calculates mean and standard deviation for all methods and all metrics contained the BenchmarkPOD object

        Parameters
        ----------
        pod : BenchmarkPOD
            data container produced by benchmarking

    """
    # register means and stds for regression
    samples = pod.get_regression_data(item='samples')
    grouped = samples.groupby(axis=1, level=['dataset', 'n_features', 'regression_metric'], sort=False)
    pod.register_regression(value=grouped.mean(), item='mean')
    pod.register_regression(value=grouped.std(), item='std')

    # register means and stds for runtime measurements
    samples = pod.get_measurement_data(item='samples')
    grouped = samples.groupby(axis=1, level=['dataset', 'n_features'], sort=False)
    pod.register_measurement(value=grouped.mean(), item='mean')
    pod.register_measurement(value=grouped.std(), item='std')


