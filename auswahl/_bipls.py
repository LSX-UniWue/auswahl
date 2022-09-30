from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from ._base import IntervalSelector


class BiPLS(IntervalSelector):
    """Feature Selection with Backward interval Partial Least Squares (BiPLS).

    The method separates the features space into intervals of equal width and sequentially removes the worst interval.
    The last interval is smaller if the total number of features is not a whole multiple of the interval width.

    The BiPLS method has been described in Xiaobo et al. [1]_.

    Read more in the :ref:`User Guide <bipls>`.

    Parameters
    ----------
    n_intervals_to_select : int, default=None
        Number of intervals to select.

    interval_width : int or float, default=None
        Number of features that form an interval.

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

    rank_ : ndarray of shape (n_features,)
        Relative rank of selection. The interval with the lowest relative rank has been removed first. The finally
        selected intervals have a relative rank of 1.

    References
    ----------
    .. [1] Zou Xiaobo, Zhao Jiewen, Li Yanxiao,
           'Selection of the efficient wavelength regions in FT-NIR spectroscopy for determination of SSC of ‘Fuji’
           apple based on BiPLS and FiPLS models',
           Vibrational Spectroscopy, vol. 44, no. 2, 220--227, 2007.

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import BiPLS
    >>> np.random.seed(1337)
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 4 * X[:,1] - 2 * X[:, 4] + 3 * X[:, 5]  # y depends on two intervals
    >>> selector = BiPLS(n_intervals_to_select=2, interval_width=2)
    >>> selector.fit(X, y).get_support()
    array([ True, True, False, False, True, True, False, False, False, False])
    """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None,
                 pls: PLSRegression = None,
                 n_cv_folds: int = 10,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 n_jobs: int = 1):
        super().__init__(n_intervals_to_select, interval_width,
                         model_hyperparams=model_hyperparams, n_cv_folds=n_cv_folds, n_jobs=n_jobs)
        self.pls = pls
        self.n_cv_folds = n_cv_folds

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        selection = np.ones(X.shape[1], dtype=bool)
        rank = np.ones(X.shape[1])
        free_idx = [i for i in range(0, X.shape[1], interval_width)]
        n_intervals_to_remove = len(free_idx) - n_intervals_to_select

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for n in range(len(free_idx) - n_intervals_to_select):
                x_free = X[:, selection]
                n_features = x_free.shape[1]
                evaluations = parallel(delayed(self.evaluate)
                                       (np.delete(x_free, np.r_[i:min(n_features, i + interval_width)], axis=1), y,
                                        self.pls)
                                       for i in range(0, n_features, interval_width))
                scores, models = list(zip(*evaluations))
                best = np.argmax(scores)
                worst_interval = free_idx[best]
                selection[worst_interval:worst_interval + interval_width] = 0
                self.best_model_ = models[best]
                free_idx.remove(worst_interval)
                rank[worst_interval:worst_interval + interval_width] = n / n_intervals_to_remove

        self.support_ = selection
        self.rank_ = rank
        return self
