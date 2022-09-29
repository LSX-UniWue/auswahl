import warnings
from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from ._base import IntervalSelector
from ._base import FeatureDescriptor


class IPLS(IntervalSelector):
    """Interval selection with Interval Partial Least Squares (iPLS).

    The optimal interval of a specified width is calculated according to Norgaard et al. [1]_.

    Read more in the :ref:`User Guide <ipls>`.

    Parameters
    ----------
    interval_width : int, default=None
        Width of the interval to select.

    n_cv_folds : int, default=10
        Number of cross validation folds used to evaluate intervals

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

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
    >>> selector.fit(X, y).get_support()
    array([False, False, False, True, True, False, False, False, False, False])
    """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None,
                 n_cv_folds: int = 10,
                 pls: PLSRegression = None,
                 n_jobs: int = 1,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 random_state: Union[int, np.random.RandomState] = None):

        super().__init__(n_intervals_to_select=1, interval_width=interval_width,
                         model_hyperparams=model_hyperparams, n_cv_folds=n_cv_folds, n_jobs=n_jobs)

        if n_intervals_to_select != 1:
            warnings.warn("""IPLS only supports the selection of a single interval. 
                             n_intervals_to_select has been clipped to 1 and the interval_width increased to 
                             n_intervals_to_select * interval_width. Hence, IPLS models the special case of aranging
                             the selected intervals as continuum.""")

        self.pls = pls
        self.random_state = random_state

    def _fit_ipls(self, X, y, interval_width, pls, start):
        score, model = self.evaluate(X[:, np.arange(start, start + interval_width, dtype='int')],
                                     y,
                                     pls)
        return score, model, start

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        candidates = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_ipls)(X,
                                                                          y,
                                                                          interval_width * n_intervals_to_select,
                                                                          self.pls,
                                                                          i) for i in range(X.shape[-1]-interval_width+1))
        score, best_model, start = max(candidates, key=lambda x: x[0])
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[start:start + interval_width] = True
        self.best_model_ = best_model

    def reparameterize(self, feature_descriptor: FeatureDescriptor):
        n_intervals_to_select, interval_width = feature_descriptor.get_configuration_for(self)
        self.interval_width = n_intervals_to_select * interval_width


