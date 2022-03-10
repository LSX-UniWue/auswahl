from typing import Union, Dict

import numpy as np
from numba import jit
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_val_score

from auswahl._base import IntervalSelector


class IPLS(IntervalSelector):

    """ Interval selection with Interal Partial Least Squares (iPLS).

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
                 interval_width: Union[int, float] = None,
                 n_cv_folds: int = 10,
                 pls: PLSRegression = None,
                 random_state: Union[int, np.random.RandomState] = None):

        super().__init__(1, interval_width)

        self.pls = pls
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state

    def _evaluate_selection(self, X, y, wavelengths, pls):
        cv_scores = cross_val_score(pls,
                                    X[:, wavelengths],
                                    y,
                                    cv=self.n_cv_folds,
                                    scoring='neg_mean_squared_error')
        return np.mean(cv_scores)

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        pls = PLSRegression() if self.pls is None else clone(self.pls)
        wavelengths = np.arange(X.shape[1], dtype='int')

        candidates = []
        for i in range(X.shape[1] - interval_width):
            candidates.append((self._evaluate_selection(X, y, wavelengths[i:i + interval_width], pls), i))

        self.score_, offset = max(candidates, key=lambda tup: tup[0])
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[offset:offset + interval_width] = True

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_


