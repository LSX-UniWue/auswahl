from typing import Union, Dict

import numpy as np
from numba import jit
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_val_score

from auswahl._base import IntervalSelector


class FOSS(IntervalSelector):

    """

        TODO

    """

    def __init__(self,
                 n_intervals_to_select: int = None,
                 interval_width: Union[int, float] = None,
                 n_cv_folds : int = 10,
                 pls: PLSRegression = None,
                 random_state: Union[int, np.random.RandomState] = None):

        super().__init__(n_intervals_to_select, interval_width)

        self.pls = pls
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state

    @jit(nopython=True)
    def _cost(self, data: np.array, left_bound, right_bound):
        return np.var(data[left_bound: right_bound + 1])

    @jit(nopython=True)
    def _build_result(self, split_points: np.array, n_groups: int):
        res = [split_points[n_groups - 1, -1]]
        index = res[0] - 1
        for i in range(n_groups - 2, 0, -1):
            res = [split_points[i, index]] + res
            index = res[0] - 1
        return res

    @jit(nopython=True)
    def _fisher_optimal_partitioning(self, data: np.array, n_groups: int):
        table = np.zeros((n_groups + 1, data.shape[0]), dtype='float')
        split_points = np.zeros_like(table, dtype='int')

        for i in range(table.shape[1]):
            group_range = min(i + 1, n_groups - 1 if (i != table.shape[1] - 1) else n_groups)

            table[0, i] = self._cost(data, 0, i)
            for j in range(1, group_range):
                costs = np.array([table[j - 1, z - 1] + self._cost(data, z, i) for z in range(j - 1, i + 1)])
                z_raw = np.argmin(costs)
                table[j, i] = costs[z_raw]
                split_points[j, i] = z_raw + (j - 1)

        return table[n_groups - 1, -1], self._build_result(split_points, n_groups)

    def _weight_variables(self, X, y, wavelengths, pls):
        x_pls_fit = X[:, wavelengths]
        pls.fit(x_pls_fit, y)
        weights = pls.coef_.flatten()# no absolute value
        return weights

    def _weight_blocks(self, variable_weights, split_points):

        blocks = np.split(np.abs(variable_weights), split_points)
        block_weights = []
        for block in blocks:
            block_weights.append(np.sum(block))
        return np.linalg.norm(block_weights, ord=1)# convert to probability distribution

    def _evaluate_selection(self, X, y, wavelengths, pls):
        cv_scores = cross_val_score(pls,
                                    X[:, wavelengths],
                                    y,
                                    cv=self.n_cv_folds,
                                    scoring='neg_mean_squared_error')
        return np.mean(cv_scores)

    def _fit(self, X, y, n_intervals_to_select, interval_width):

        pls = PLSRegression() if self.pls is None else clone(self.pls)
        random_state = check_random_state(self.random_state)
        wavelengths = np.arange(X.shape[1], dtype='int')

        opt_wavelengths = None
        opt_score = -1e10

        for i in range(100):# TODO: devise a reasaonable concept here

            weights = self._weight_variables(X, y, wavelengths, pls)
            _, split_points = self._fisher_optimal_partitioning(weights, 10)# TODO: devise a reasonable concept here
            block_weights = self._weight_blocks(weights, split_points)
            wavelength_blocks = np.split(wavelengths, split_points)

            sampled_blocks = random_state.choice(np.arange(len(block_weights)),
                                                 10,# TODO: devisa a reasaonable concept here
                                                 replace=True,
                                                 p=block_weights)
            selected_blocks = np.sort(np.unique(sampled_blocks))

            wavelengths = np.concatenate([wavelength_blocks[z] for z in selected_blocks])

            score = self._evaluate_selection(X, y, wavelengths, pls)

            if score > opt_score:
                opt_wavelengths = wavelengths
                opt_score = score

        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[opt_wavelengths] = True

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_


