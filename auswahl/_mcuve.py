from typing import Union, List, Dict

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, check_scalar

from .util import get_coef_from_pls
from ._base import PointSelector


class MCUVE(PointSelector):
    """Feature selection with Monte Carlo Uninformative Variable Elimination.

    The stability for each feature is computed according to Cai et al. [1]_.

    Note that the **absolute** stability values are used to determine the most important features.

    Read more in the :ref:`User Guide <mcuve>`.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        Number of features to select.

    n_subsets : int, default=100
        Number of random subsets to create.

    n_samples_per_subset : int or float, default=None
        Number of samples used for each random subset.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

    random_state : int or numpy.random.RandomState, default=None
        Seed for the random subset sampling. Pass an int for reproducible output across function calls.

    Attributes
    ----------
    coefs_ : ndarray of shape (n_subsets, n_features)
        Fitted regression coefficients of the <n_subsets> PLS models.

    stability_ : ndarray of shape (n_features,)
        Computed stability value for each feature. While these attribute contains the signed stability values, MC-UVE
        uses the absolute values to select the most important features.

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
    >>> selector.fit(X, y).get_support()
    array([ True, False, False, False, False, True, False, False, False, False])
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 n_subsets: int = 100,
                 n_samples_per_subset: Union[int, float] = None,
                 pls: PLSRegression = None,
                 n_cv_folds: int = 5,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 random_state: Union[int, np.random.RandomState] = None):
        super().__init__(n_features_to_select, model_hyperparams, n_cv_folds, random_state)
        self.n_subsets = n_subsets
        self.n_samples_per_subset = n_samples_per_subset
        self.pls = pls

    def _fit(self, X, y, n_features_to_select):
        _, model = self.evaluate(X, y, self.pls, do_cv=False)
        random_state = check_random_state(self.random_state)
        self._check_n_subsets()
        n_samples_per_subset = self._check_n_samples_per_subset(X)

        n_samples = X.shape[0]
        coefs = []
        for i in range(self.n_subsets):
            idx = random_state.permutation(n_samples)[:n_samples_per_subset]
            X_i, y_i = X[idx], y[idx]

            model.fit(X_i, y_i)
            coefs.append(get_coef_from_pls(model).squeeze())

        self.coefs_ = np.array(coefs)
        self.stability_ = self.coefs_.mean(axis=0) / self.coefs_.std(axis=0)

        selected_idx = np.argsort(abs(self.stability_))[-n_features_to_select:]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[selected_idx] = 1
        _, self.best_model_ = self.evaluate(X[:, self.support_], y, self.pls, do_cv=False)

        return self

    def _check_n_subsets(self):
        check_scalar(x=self.n_subsets, name='n_subsets', target_type=int, min_val=2)

    def _check_n_samples_per_subset(self, X):
        n_samples = X.shape[0]
        n_samples_per_subset = self.n_samples_per_subset

        if n_samples_per_subset is None:
            n_samples_per_subset = n_samples // 2
        elif 0 < n_samples_per_subset < 1:
            n_samples_per_subset = max(1, int(n_samples_per_subset * n_samples))

        if (n_samples_per_subset <= 0) or (n_samples_per_subset >= n_samples):
            raise ValueError('n_samples_per_subset has to be either an int in {1, ..., n_samples-1}'
                             f'or a float in (0, 1); got {self.n_samples_per_subset}')

        return n_samples_per_subset
