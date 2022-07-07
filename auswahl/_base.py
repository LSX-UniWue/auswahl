from abc import ABCMeta, abstractmethod
from typing import Union, List

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_scalar


class CVEvaluator:

    def __init__(self, model_hyperparams, n_cv_folds):
        self.model_hyperparams = model_hyperparams
        if self.model_hyperparams is not None and not isinstance(self.model_hyperparams, (list, dict)):
            raise ValueError("Keyword argument 'model_hyperparams' is expected to be of type dict or list of dicts")
        self.n_cv_folds = n_cv_folds
        if not isinstance(self.n_cv_folds, int) or self.n_cv_folds <= 0:
            raise ValueError(f'Keyword argument "n_cv_folds" is expected to be a positive integer. Got {self.cv_folds}')

    def _evaluate(self, X, y, model, do_cv=True):
        model = PLSRegression(n_components=min(2, X.shape[1])) if model is None else clone(model)
        if self.model_hyperparams is None:  # no hyperparameter optimization; conduct a simple CV
            cv_scores = None
            if do_cv:
                cv_scores = np.mean(cross_val_score(model, X, y, cv=self.n_cv_folds, scoring='neg_mean_squared_error'))
            model.fit(X, y)
            return cv_scores, model
        else:
            cv = GridSearchCV(model, self.model_hyperparams, cv=self.n_cv_folds, scoring='neg_mean_squared_error')
            cv.fit(X, y)
            return cv.best_score_, cv.best_estimator_


class PointSelector(CVEvaluator, SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for feature selection methods that select features independently.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        Number of features to select
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 model_hyperparams: Union[dict, List[dict]] = None,
                 n_cv_folds: int = 2):
        self.n_features_to_select = n_features_to_select
        super().__init__(model_hyperparams, n_cv_folds)

    def fit(self, X, y, mask=None):
        """Run the feature selection process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        mask: np.array of shape (n_features,)
            Mask indicating (values == 0), which features are not to be taken into account during the feature selection

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if mask is not None:
            if mask.shape != (X.shape[1],):
                raise ValueError(f'Expected mask to have shape {(X.shape[1],)}. Got {mask.shape}')
            mask_indices = np.nonzero(mask)[0]
            n_features = X.shape[1]
            X = X[:, mask_indices]

        X, y = self._validate_data(X, y, accept_sparse=False, ensure_min_samples=2, ensure_min_features=2)
        n_features_to_select = self._check_n_features_to_select(X)
        self._fit(X, y, n_features_to_select)

        if mask is not None:
            selected = mask_indices[np.nonzero(self.support_)]
            self.support_ = np.zeros((n_features,), dtype=bool)
            self.support_[selected] = 1

        return self

    @abstractmethod
    def _fit(self, X, y, n_features_to_select):
        pass

    def _check_n_features_to_select(self, X):
        n_features = X.shape[1]
        n_features_to_select = self.n_features_to_select

        if n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            check_scalar(n_features_to_select,
                         name='n_features_to_select',
                         target_type=(int, float))

        if 0 < n_features_to_select < 1:
            n_features_to_select = int(n_features_to_select * n_features)

        if (n_features_to_select <= 0) or (n_features_to_select >= n_features):
            raise ValueError('n_features_to_select has to be either an int in {1, ..., n_features-1}'
                             'or a float in (0, 1) with (n_features_to_select*n_features) >= 1; '
                             f'got {self.n_features_to_select}')

        return n_features_to_select

    def set_n_features(self, n_features):
        self.n_features_to_select = n_features


class IntervalSelector(CVEvaluator, SelectorMixin, BaseEstimator, metaclass=ABCMeta):
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
                 interval_width: Union[int, float] = None, n_cv_folds: int = 1,
                 model_hyperparams: Union[dict, List[dict]] = None):
        self.n_intervals_to_select = n_intervals_to_select
        self.interval_width = interval_width
        super().__init__(model_hyperparams, n_cv_folds)

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
        self._check_n_intervals_to_select(X)
        interval_width = self._check_interval_width(X)

        self._fit(X, y, self.n_intervals_to_select, interval_width)
        return self

    @abstractmethod
    def _fit(self, X, y, n_intervals_to_select, interval_width):
        pass

    def _check_n_intervals_to_select(self, X):
        check_scalar(self.n_intervals_to_select,
                     name='n_intervals_to_select',
                     target_type=int, min_val=1,
                     max_val=X.shape[1]-1)

    def _check_interval_width(self, X):
        n_features = X.shape[1]
        interval_width = self.interval_width

        if interval_width is None:
            interval_width = n_features // 2
        elif 0 < interval_width < 1:
            interval_width = max(2, int(interval_width * n_features))

        if (interval_width <= 0) \
                or (interval_width >= n_features) \
                or (self.n_intervals_to_select * interval_width >= n_features):

            raise ValueError('interval_width has to be either an int in {1, ..., n_features-1}'
                             f'or a float in (0, 1); got {self.interval_width}')

        return interval_width

    def set_interval_params(self, n_intervals, interval_width):
        self.interval_width = interval_width
        self.n_intervals_to_select = n_intervals
