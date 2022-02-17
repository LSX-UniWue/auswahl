from typing import Union, Dict

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from auswahl._base import PointSelector


class VIP(PointSelector):
    """Feature Selection with Variable Importance in Projection.

    The VIP scores are computed according to Favilla et al. [1]_.

    Parameters
    ----------
    n_features_to_select: int or float, default=None
        Number of features to select.
    pls_kwargs: dictionary
        Keyword arguments that are passed to :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>`.

    Attributes
    ----------
    pls_estimator_: PLSRegression instance
        Fitted PLS estimator used to calculate the vip scores.
    vips_: ndarray of shape (n_features,)
        Calculated VIP scores.
    support_ : ndarray of shape (n_features,)
        Mask of selected features.

    References
    ----------
    .. [1] Stefania Favilla, Caterina Durante, Mario Li Vigni, Marina Cocchi,
           'Assessing feature relevance in NPLS models by VIP',
           Chemometrics and Intelligent Laboratory Systems, 129, 76--86, 2013.

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import VIP
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
    >>> selector = VIP(n_features_to_select=2)
    >>> selector.fit(X, y)
    >>> selector.get_support()
    array([True, False, False, False, False, True, False, False, False, False])
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 pls_kwargs: Dict = None):
        super().__init__(n_features_to_select)
        self.pls_kwargs = pls_kwargs

    def _fit(self, X, y, n_features_to_select):
        pls_kwargs = dict() if self.pls_kwargs is None else self.pls_kwargs
        self.pls_estimator_ = PLSRegression(**pls_kwargs)
        self.pls_estimator_.fit(X, y)
        self.vips_ = self._calculate_vip_scores(X)

        selected_idx = np.argsort(self.vips_)[-n_features_to_select:]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[selected_idx] = 1

        return self

    def _calculate_vip_scores(self, X):
        x_scores = self.pls_estimator_.transform(X)
        x_weights = self.pls_estimator_.x_weights_
        y_loadings = self.pls_estimator_.y_loadings_

        num_features = X.shape[1]
        total_explained_variance = np.diag((x_scores.T @ x_scores) @ (y_loadings.T @ y_loadings))[:, None]

        x_weights_normalized = (x_weights / np.linalg.norm(x_weights, axis=0, keepdims=True)) ** 2
        explained_variance = x_weights_normalized @ total_explained_variance
        vips = np.sqrt((num_features * explained_variance) / total_explained_variance.sum())

        return vips.flatten()

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_
