from abc import ABCMeta, abstractmethod
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_scalar


class PointSelector(SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for feature selection methods that select features independently.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        Number of features to select
    """

    def __init__(self, n_features_to_select: Union[int, float] = None):
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        """Run the feature selection process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
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


class IntervalSelector(SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for feature selection methods that select consecutive chunks (intervals) of features.

    Parameters
    ----------
    n_intervals_to_select : int, default=None
        Number of intervals to select.

    interval_width : int or float, default=None
        Number of features that form an interval
    """

    def __init__(self,
                 n_intervals_to_select: int = None,
                 interval_width: Union[int, float] = None):
        self.n_intervals_to_select = n_intervals_to_select
        self.interval_width = interval_width

    def fit(self, X, y):
        """Run the feature selection process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._validate_data(X, y, accept_sparse=False, ensure_min_samples=2, ensure_min_features=2)
        self._check_n_intervals_to_select()
        interval_width = self._check_interval_width(X)

        self._fit(X, y, self.n_intervals_to_select, interval_width)
        return self

    @abstractmethod
    def _fit(self, X, y, n_intervals_to_select, interval_width):
        pass

    def _check_n_intervals_to_select(self):
        check_scalar(self.n_intervals_to_select, name='n_intervals_to_select', target_type=int, min_val=1)

    def _check_interval_width(self, X):
        n_features = X.shape[1]
        interval_width = self.interval_width

        if interval_width is None:
            interval_width = n_features // 2
        elif 0 < interval_width < 1:
            interval_width = max(2, int(interval_width * n_features))

        if (interval_width <= 0) or (interval_width >= n_features):
            raise ValueError('interval_width has to be either an int in {1, ..., n_features-1}'
                             f'or a float in (0, 1); got {self.interval_width}')

        return interval_width
