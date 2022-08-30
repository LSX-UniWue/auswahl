import warnings

from sklearn.cross_decomposition import PLSRegression


def get_coef_from_pls(pls):
    """Retrieves the coef_ attribute from the PLS model in the shape (n_targets, n_features) without triggering a
    FutureWarning. The coef_ values are already in the planned shape for future releases.

    Parameters
    ----------
    pls: PLSRegression
        Fitted Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class.

    Returns
    -------
    coef_: ndarray of shape (n_targets, n_features)
        Linear coefficients of the pls model
    """
    warnings.simplefilter('ignore', category=FutureWarning)
    coef = pls.coef_
    warnings.simplefilter('default', category=FutureWarning)
    return coef.T
