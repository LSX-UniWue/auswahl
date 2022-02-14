from typing import Union, Dict

import numpy as np
from sklearn.cross_decomposition import PLSCanonical
from sklearn.utils.validation import check_is_fitted

from auswahl._base import PointSelector


class VIP(PointSelector):
    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 pls_kwargs: Dict = None):
        super().__init__(n_features_to_select)
        self.pls_kwargs = pls_kwargs

    def _fit(self, X, y, n_features_to_select):
        pls_kwargs = dict() if self.pls_kwargs is None else self.pls_kwargs
        self.pls_estimator_ = PLSCanonical(**pls_kwargs)
        self.pls_estimator_.fit(X, y)
        self.vips_ = self._calculate_vip_scores(X)

        selected_idx = np.argsort(self.vips_)[-n_features_to_select:]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[selected_idx] = 1

        return self

    def _calculate_vip_scores(self, X):
        x_scores = np.dot(X, self.pls_estimator_.x_rotations_)
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
