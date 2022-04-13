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
