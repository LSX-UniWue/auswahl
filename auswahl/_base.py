from abc import ABCMeta, abstractmethod
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin


class PointSelector(SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for feature selection methods that select features independently.

    Parameters
    ----------
    n_features_to_select: int or float, default=None
        Number of features to select
    """

    def __init__(self, n_features_to_select: Union[int, float] = None):
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        """Run the feature selection process.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.
        y: array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        X, y = self._validate_data(X, y, accept_sparse=False, ensure_min_samples=2, ensure_min_features=2)
        n_features_to_select = self._check_n_features_to_select(X)
        self._fit(X, y, n_features_to_select)
        return self

    @abstractmethod
    def _fit(self, X, y, n_features_to_select):
        pass

    def _check_n_features_to_select(self, X):
        n_features = X.shape[1]
        n_features_to_select = self.n_features_to_select

        if n_features_to_select is None:
            n_features_to_select = n_features // 2
        if 0 < n_features_to_select < 1:
            n_features_to_select = int(n_features_to_select * n_features)

        if (n_features_to_select <= 0) or (n_features_to_select >= n_features):
            raise ValueError('n_features_to_select has to be either an int in {1, ..., n_features-1}'
                             'or a float in (0, 1) with (n_features_to_select*n_features) >= 1; '
                             f'got {self.n_features_to_select}')

        return n_features_to_select
