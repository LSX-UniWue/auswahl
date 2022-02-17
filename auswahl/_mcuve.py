from typing import Union, Dict

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from auswahl._base import PointSelector


class MCUVE(PointSelector):
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
