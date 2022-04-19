import numpy as np


def selection_stability_score(supports: list):
    """
        Calculates a selection stability score for randomized selection methods.
        If r is the number of runs, n the number of features to select and d the number of distinct features selected
        across the runs, the score in range [0,1] is calculated as:

                        score = (n - (d - n)/(r -1)) / n

        Parameters
        ----------
        supports : list of arrays
            list of arrays of indices of features selected in each run

        Returns
        -------
        stability score : float
            stability score calcuated according to the above formula

    """

    n = supports[0].shape[0]
    d = np.distinct(np.concatenate(supports)).shape[0]
    r = len(supports)

    return (n - (d - n)/(r - 1)) / n


def _intersection_expecatation(n, s):
    logs = np.log(np.arange(1, n + 1))
    fix_summand = 2 * np.sum(logs[:s]) + 2 * np.sum(logs[:n - s]) - np.sum(logs)

    m_fac = - logs[0]
    sm_fac = -2 * np.sum(logs[:s - 1])
    nssm_fac = - np.sum(logs[:n - 2 * s + 1])

    e = 0
    for i in range(s - 1):
        e += np.exp(m_fac + sm_fac + nssm_fac + logs[i] + fix_summand)
        m_fac -= logs[i + 1]
        sm_fac += 2 * logs[s - i - 2]
        nssm_fac -= logs[n - 2 * s + i + 1]

    # special case intersection of size s
    e += np.exp(logs[s - 1] + np.sum(logs[:s]) + np.sum(logs[:n - s]) - np.sum(logs))
    return e


# todo: nomenclature!
def stability_score(supports: list, n_wavelengths: int):
    """
            Calculates a selection stability score for randomized selection methods, according to Deng et al. [1]_.

            Parameters
            ----------
            supports : list of arrays
                list of arrays of indices of features selected in each run

            n_wavelengths : int
                number of wavelengths for the regression problem at hand

            Returns
            -------
            stability score : float
                stability score calcuated according to Deng et al. [1]_

            References
            ----------
            .. [1] Bai-Chuan Deng, Yong-Huan Yun, Pan Ma, Chen-Chen Li, Da-Bing Ren and Yi-Zeng Liang,
                   'A new method for wavelength interval selection that intelligently optimizes the locations, widths
                   and combination of intervals',
                   Analyst, 6, 1876-1885, 2015.

    """
    r = len(supports)
    pairwise_sim = np.empty((r**2 - r) / 2, dtype='float')

    sample_size = supports[0].shape[0]
    e = _intersection_expecatation(n_wavelengths, sample_size)
    for i, (x, y) in enumerate(np.triu_indices(r)):
        if x != y:
            pairwise_sim[i] = (np.intersect1d(supports[x, y]) - e) / (np.sqrt(supports[x].shape[0] * supports[y].shape[0])
                                                                      - e)
    return np.sum(pairwise_sim) / (2 / (r*(r - 1)))

