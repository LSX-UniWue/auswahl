from typing import Union, List, Dict

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from ._base import IntervalSelector


class FiPLS(IntervalSelector):
    """Feature Selection with Forward interval Partial Least Squares (FiPLS).

    The FiPLS method has been described in Xiaobo et al. [1]_.
    This implementation deviates from the original description as it allows to select intervals at arbitrary positions.

    Read more in the :ref:`User Guide <fipls>`.

    Parameters
    ----------
    n_intervals_to_select : int, default=None
        Number of intervals to select.

    interval_width : int or float, default=None
        Number of features that form an interval

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

    n_cv_folds : int, default=10
        Number of cross validation folds used to evaluate intervals

    n_jobs : int, default=1
        Number of parallel processes that fit PLS models on the different intervals

    Attributes
    ----------
    support_ : ndarray of shape (n_features,)
        Mask of selected features.

    References
    ----------
    .. [1] Zou Xiaobo, Zhao Jiewen, Li Yanxiao,
           'Selection of the efficient wavelength regions in FT-NIR spectroscopy for determination of SSC of ‘Fuji’
           apple based on BiPLS and FiPLS models',
           Vibrational Spectroscopy, vol. 44, no. 2, 220--227, 2007.

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import FiPLS
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 4 * X[:,1] - 2 * X[:, 5] + 3 * X[:,6]  # y depends on two intervals
    >>> selector = FiPLS(n_intervals_to_select=2, interval_width=2)
    >>> selector.fit(X, y).get_support()
    array([ True, True, False, False, False, True, True, False, False, False])
    """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None,
                 pls: PLSRegression = None,
                 n_cv_folds: int = 10,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 n_jobs: int = 1):
        super().__init__(n_intervals_to_select, interval_width,
                         n_cv_folds=n_cv_folds, model_hyperparams=model_hyperparams, n_jobs=n_jobs)
        self.pls = pls
        self.n_cv_folds = n_cv_folds

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        selection = np.zeros(X.shape[1], dtype=bool)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for n in range(n_intervals_to_select):
                x_selected = X[:, selection]
                x_free = X[:, ~selection]
                free_idx = np.arange(X.shape[1])[~selection]
                evaluations = parallel(delayed(self.evaluate)
                                  (np.concatenate([x_selected, x_free[:, i:i + interval_width]], axis=1), y, self.pls)
                                  for i in range(len(free_idx) - interval_width + 1))
                scores, models = list(zip(*evaluations))
                best_idx = np.argmax(scores)
                selection[free_idx[best_idx]:free_idx[best_idx + interval_width - 1] + 1] = 1
                self.best_model_ = models[best_idx]

        self.support_ = selection
        return self
