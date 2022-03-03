from typing import Union
from warnings import warn

import numpy as np
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from auswahl._base import PointSelector


class VIP(PointSelector):
    """Feature Selection with Variable Importance in Projection.

    The VIP scores are computed according to Favilla et al. [1]_.

    Read more in the :ref:`User Guide <vip>`.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        Number of features to select.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class.
        Use this to adjust the hyperparameters of the PLS method.

    Attributes
    ----------
    pls_ : PLSRegression instance
        Fitted PLS estimator used to calculate the VIP scores.

    vips_ : ndarray of shape (n_features,)
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
                 pls: PLSRegression = None):
        super().__init__(n_features_to_select)
        self.pls = pls

    def _fit(self, X, y, n_features_to_select):
        self.pls_ = PLSRegression() if self.pls is None else clone(self.pls)
        self.pls_.fit(X, y)
        self.vips_ = self._calculate_vip_scores(X)

        selected_idx = np.argsort(self.vips_)[-n_features_to_select:]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[selected_idx] = 1

        return self

    def _calculate_vip_scores(self, X):
        x_scores = self.pls_.transform(X)
        x_weights = self.pls_.x_weights_  # already normalized
        y_loadings = self.pls_.y_loadings_

        num_features = X.shape[1]
        explained_variance = (y_loadings ** 2) @ (x_scores.T @ x_scores)
        weighted_explained_variance = (x_weights ** 2) @ explained_variance.T
        vips = np.sqrt((num_features * weighted_explained_variance) / explained_variance.sum())

        return vips.flatten()

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def get_support_for_threshold(self, threshold: float = 1, indices: bool = False):
        """Select a set of features whose VIP values are above a given threshold.

        Parameters
        ----------
        threshold : float, default=1
            Lower bound that has to be exceeded by the VIP value of a feature so that it is selected.

        indices : bool, default=False
            If True, the return value will be an array of integers, rather than a boolean mask.

        Returns
        -------
        selection : ndarray of shape (n_features,)
            Boolean mask of selected features, or array of indices if indices=True.
        """
        check_is_fitted(self)
        mask = self.vips_ > threshold
        if not np.any(mask):
            warn(f'No VIP score is higher than the given threshold of {threshold}. '
                 f'Only the most important feature will be selected with a VIP value of {self.vips_.max()}')
            mask[np.argmax(self.vips_)] = 1

        return mask if not indices else np.where(mask)[0]
