
from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Union, Tuple, List
from functools import cached_property

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_scalar
from sklearn.utils.validation import check_is_fitted



class FeatureDescriptor:

    """
        TODO: all docstrings
    """

    def __init__(self, key: Union[int, Tuple[int, int], FeatureDescriptor], resolve_tuples: bool = False):
        if isinstance(key, FeatureDescriptor):
            self._build_from_descriptor(key)
        else:
            self._check_consistency(key)
            self.org_key = key
            self.key = self._resolve_tuples(key, resolve_tuples)
            self.resolve_tuples = resolve_tuples

    def _build_from_descriptor(self, descriptor):
        self.key = descriptor.key
        self.org_key = descriptor.org_key
        self.resolve_tuples = descriptor.resolve_tuples

    def __len__(self):
        return self.comparator[0]

    @cached_property
    def string_rep(self):
        if isinstance(self.key, int):
            return str(self.key)
        return f'{self.key[0]}/{self.key[1]}'

    @cached_property
    def comparator(self):
        if isinstance(self.key, int):
            return [self.key]
        return [self.key[0] * self.key[1], self.key[0], self.key[1]]

    #
    # Consistency checks
    #

    def _check_consistency(self, x):
        if isinstance(x, int):
            self._check_positive_integer(x)
        elif isinstance(x, tuple):
            self._check_diploid_positive_integer_tuple(x)
        else:
            raise ValueError(f'The specification of features requires either a positive integer'
                             f' or a tuple of two positive integers')

    def _check_positive_integer(self, x):
        if not isinstance(x, int):
            raise ValueError(f'The specification of features requires integers. Got {type(x)}')
        if x <= 0:
            raise ValueError(f'The specification of features requires positive integers. Got {x}')

    def _check_diploid_positive_integer_tuple(self, x):
        if len(x) != 2:
            raise ValueError("Feature specification with tuples requires a tuple of length 2.")
        v1, v2 = x
        self._check_positive_integer(v1)
        self._check_positive_integer(v2)
        return x

    def _resolve_tuples(self, key, resolve_tuple):
        if resolve_tuple and isinstance(key, tuple):
            return key[0] * key[1]  # consistency has already been checked at this point
        return key

    #
    # Make the class hashable
    #
    def __hash__(self):
        return self.key.__hash__()

    #
    # Comparator implementations
    #
    def __le__(self, x: FeatureDescriptor):
        for i in range(len(self.comparator)):
            if self.comparator[i] < x.comparator[i]:
                return True
            elif self.comparator[i] > x.comparator[i]:
                return False
        return True

    def __ge__(self, x: FeatureDescriptor):
        for i in range(len(self.comparator)):
            if self.comparator[i] > x.comparator[i]:
                return True
            elif self.comparator[i] < x.comparator[i]:
                return False
        return True

    #
    # Comparison arbitrage section
    #

    def __eq__(self, x: FeatureDescriptor):
        return self.__le__(x) and self.__ge__(x)

    def __gt__(self, x: FeatureDescriptor):
        return not self.__le__(x)

    def __ne__(self, x: FeatureDescriptor):
        return not self.__eq__(x)

    def __lt__(self, x: FeatureDescriptor):
        return not self.__ge__(x)

    #
    # Printing
    #
    def __repr__(self):
        return self.string_rep

    def __str__(self):
       return self.string_rep

    #
    # Interface for selectors
    #
    def get_params_for(self, selector: SpectralSelector):
        if isinstance(selector, PointSelector):
            if self.resolve_tuples:
                return self.key
            else:
                return self.key[0] * self.key[1]
        else:
            return self.key[0], self.key[1]


class Selection:

    """
        TODO
    """

    def __init__(self, selected_features: List = None):
        self.selected_features = selected_features

    def __repr__(self):
        return str(self.selected_features)

    def __str__(self):
        return self.__repr__()


class SpectralSelector(SelectorMixin, BaseEstimator, metaclass=ABCMeta):

    """
        TODO
    """

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
        self._dispatch_fit(X, y)

        if mask is not None:
            selected = mask_indices[np.nonzero(self.support_)]
            self.support_ = np.zeros((n_features,), dtype=bool)
            self.support_[selected] = 1

        return self

    def get_best_estimator(self):
        """
            TODO
        """
        check_is_fitted(self)
        if not hasattr(self, 'best_model_'):
            raise NotImplementedError("Make sure, that after fit has been called on the selector, the selector "
                                      "provides the optimally configured estimator for the selected features as "
                                      "attribute 'best_model_'")
        return self.best_model_

    @abstractmethod
    def _dispatch_fit(self, X, y):
        ...

    @abstractmethod
    def reparameterize(self, feature_descriptor: FeatureDescriptor):
        ...


class PointSelector(SpectralSelector, metaclass=ABCMeta):
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

    def _dispatch_fit(self, X, y):
        n_features_to_select = self._check_n_features_to_select(X)
        self._fit(X, y, n_features_to_select)

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

    def reparameterize(self, feature_descriptor: FeatureDescriptor):
        self.n_features_to_select = feature_descriptor.get_params_for(self)


class IntervalSelector(SpectralSelector, metaclass=ABCMeta):
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

    def _dispatch_fit(self, X, y):
        self._check_n_intervals_to_select(X)
        interval_width = self._check_interval_width(X)
        self._fit(X, y, self.n_intervals_to_select, interval_width)

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

    def reparameterize(self, feature_descriptor: FeatureDescriptor):
        self.n_intervals_to_select, self.interval_width = feature_descriptor.get_params_for(self)
