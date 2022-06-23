from typing import Union, Dict

import warnings
import numpy as np
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score

from joblib import Parallel, delayed

from auswahl._base import IntervalSelector


class IPLS(IntervalSelector):

    """ Interval selection with Interval Partial Least Squares (iPLS).

        The optimal interval of a specified width is calculated according to Norgaard et al. [1]_.

        Parameters
        ----------
        interval_width : int, default=None
            Width of the interval to select.

        n_cv_folds : int, default=10
            Number of cross validation folds used to evaluate intervals

        pls : PLSRegression, default=None
            Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class.
            Use this to adjust the hyperparameters of the PLS method.

        Attributes
        ----------
        support_ : ndarray of shape (n_features,)
            Mask of the selected interval.

        score_ : float
            Cross validation score of the interval selected.

        References
        ----------
        .. [1] L. Nogaard, A. Saudland, J. Wagner, J. P. Nielsen, L. Munck, S. B. Engelsen,
               'Interval Partial Least-Squares Regression (iPLS):
               A comparative chemometric study with an example from Near-Infrared Spectrocopy'
               Applied Spectrosopy, Volume 54, Nr. 3, 413--419, 2000.

        Examples
        --------
        >>> import numpy as np
        >>> from auswahl import IPLS
        >>> X = np.random.randn(100, 10)
        >>> y = 5 * X[:, 3] - 2 * X[:, 4]  # y only depends on two features
        >>> selector = IPLS(interval_width=2)
        >>> selector.fit(X, y)
        >>> selector.get_support()
        array([False, False, False, True, True, False, False, False, False, False])
        """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None,
                 n_cv_folds: int = 10,
                 pls: PLSRegression = None,
                 n_jobs: int = 1,
                 random_state: Union[int, np.random.RandomState] = None):

        super().__init__(n_intervals_to_select=1, interval_width=interval_width * n_intervals_to_select)

        if n_intervals_to_select != 1:
            warnings.warn("""IPLS only supports the selection of a single interval. 
                             n_intervals_to_select has been clipped to 1 and the interval_width increased to 
                             n_intervals_to_select * interval_width. Hence, IPLS models the special case of aranging
                             the selected intervals as continuum.""")

        self.pls = pls
        self.n_cv_folds = n_cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _evaluate_selection(self, X, y, wavelengths, pls):
        cv_scores = cross_val_score(pls,
                                    X[:, wavelengths],
                                    y,
                                    cv=self.n_cv_folds,
                                    scoring='neg_mean_squared_error')
        return np.mean(cv_scores)

    def _fit_ipls(self, X, y, interval_width, pls, start):
        pls = PLSRegression() if pls is None else clone(pls)
        score = self._evaluate_selection(X,
                                         y,
                                         np.arange(start, start + interval_width, dtype='int'),
                                         pls)
        return score, start

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        candidates = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_ipls)(X,
                                                                          y,
                                                                          interval_width,
                                                                          self.pls,
                                                                          i) for i in range(X.shape[-1]-interval_width+1))
        score, start = max(candidates, key=lambda x: x[0])
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[start:start + interval_width] = True

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def set_interval_params(self, n_intervals, interval_width):
        self.interval_width = n_intervals * interval_width


