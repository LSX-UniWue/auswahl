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

        X, y = self._validate_data(X, y, accept_sparse=False, ensure_min_features=2)
        n_features_to_select = self._check_n_features_to_select(X)
        self._fit(X, y, n_features_to_select)
        return self

    @abstractmethod
    def _fit(self, X, y, n_features_to_select):
        pass

    def _check_n_features_to_select(self, X):
        n_features = X.shape[1]

        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        elif self.n_features_to_select < 0:
            raise ValueError('n_feature_to_select has to be an integer or float > 0!')
        elif isinstance(self.n_features_to_select, int):
            n_features_to_select = self.n_features_to_select
        elif self.n_features_to_select > 1.0:
            raise ValueError('If n_features_to_select is a float, it has to be in (0, 1]')
        else:
            n_features_to_select = int(n_features * self.n_features_to_select)

        return n_features_to_select
