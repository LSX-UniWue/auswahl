from typing import Union, Dict

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from auswahl._base import PointSelector


class MCUVE(PointSelector):
    """Feature selection with Monte Carlo Uninformative Variable Elimination.

    The stability for each feature is computed according to Cai et al. [1]_.
    While not stated in the original publication, we use the absolute values of the regression coefficients for
    computing the stability values.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        Number of features to select.

    n_subsets : int, default=100
        Number of random subsets to create.

    n_samples_per_subset : int or float, default=None
        Number of samples used for each random subset.

    pls_kwargs : dictionary, default=None
        Keyword arguments that are passed to :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>`.

    random_state : int or numpy.random.RandomState, default=None
        Seed for the random subset sampling. Pass an int for reproducible output across function calls.

    Attributes
    ----------
    coefs_ : ndarray of shape (n_subsets, n_features)
        Fitted regression coefficients of the <n_subsets> PLS models.

    stability_ : ndarray of shape (n_features,)
        Computed stability score of the absolute regression coefficients.

    support_ : ndarray of shape (n_features,)
        Mask of selected features.

    References
    ----------
    .. [1] Wensheng Cai, Yankun Li and Xueguang Shao,
           'A variable selection method based on uninformative variable elimination for
           multivariate calibration of near-infrared spectra',
           Chemometrics and Intelligent Laboratory Systems, 90, 188-194, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import MCUVE
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
    >>> selector = MCUVE(n_features_to_select=2)
    >>> selector.fit(X, y)
    >>> selector.get_support()
    array([True, False, False, False, False, True, False, False, False, False])
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 n_subsets: int = 100,
                 n_samples_per_subset: Union[int, float] = None,
                 pls_kwargs: Dict = None,
                 random_state: Union[int, np.random.RandomState] = None):
        super().__init__(n_features_to_select)
        self.n_subsets = n_subsets
        self.n_samples_per_subset = n_samples_per_subset
        self.pls_kwargs = pls_kwargs
        self.random_state = random_state

    def _fit(self, X, y, n_features_to_select):
        pls_kwargs = dict() if self.pls_kwargs is None else self.pls_kwargs
        random_state = check_random_state(self.random_state)
        self._check_n_subsets()
        n_samples_per_subset = self._check_n_samples_per_subset(X)

        n_samples = X.shape[0]
        coefs = []
        for i in range(self.n_subsets):
            idx = random_state.permutation(n_samples)[:n_samples_per_subset]
            X_i, y_i = X[idx], y[idx]

            pls_i = PLSRegression(**pls_kwargs)
            pls_i.fit(X_i, y_i)
            coefs.append(abs(pls_i.coef_.flatten()))
        self.coefs_ = np.array(coefs)
        self.stability_ = self.coefs_.mean(axis=0) / self.coefs_.std(axis=0)

        selected_idx = np.argsort(self.stability_)[-n_features_to_select:]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[selected_idx] = 1

        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def _check_n_subsets(self):
        if self.n_subsets < 2:
            raise ValueError(f'n_subsets has to be a positive integer >= 2; got {self.n_subsets}')

    def _check_n_samples_per_subset(self, X):
        n_samples = X.shape[0]
        n_samples_per_subset = self.n_samples_per_subset

        if n_samples_per_subset is None:
            n_samples_per_subset = n_samples // 2
        elif 0 < n_samples_per_subset < 1:
            n_samples_per_subset = int(n_samples_per_subset * n_samples)

        if (n_samples_per_subset <= 0) or (n_samples_per_subset >= n_samples):
            raise ValueError('n_samples_per_subset has to be either an int in {1, ..., n_samples-1}'
                             'or a float in (0, 1) with (n_samples_per_subset*n_samples) >= 1; '
                             f'got {self.n_samples_per_subset}')

        return n_samples_per_subset
