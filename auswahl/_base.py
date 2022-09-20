
from __future__ import annotations

import numpy as np

from abc import ABCMeta, abstractmethod

from typing import Union, Tuple, List
from functools import cached_property

import sklearn.base
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_scalar
from sklearn.utils.validation import check_is_fitted
from numpy.random import RandomState

from functools import wraps


class FeatureDescriptor:
    """ The class FeatureDescriptor abstracts the configuration of features the selection methods are to retrieve from
    the spectral data. The FeatureDescriptor wraps either a number of arbitrary features to be selected or a specific
    number of intervals of features of a fix length.

    Parameters
    ----------
    key: int, Tuple[int, int], FeatureDescriptor
        Feature configuration to be abstracted by the object. A single integer is interpreted as a number of arbitrarily
        selectable features. A tuple is a (#intervals, width of intervals) configuration of features to be selected. If a
        FeatureDescriptor is passed, it is copied. All passed integers are required to be non-negative.

    resolve_intervals: bool, default=False
        Flag indicating whether interval feature configurations are to be resolved to a single integer of arbitrary
        features to be selected.

    Attributes
    ----------
    org_key: int, Tuple[int, int]
        Originally passed feature configuration

    key: int, Tuple[int, int]
        Resolved key. Equal to org_key, if org_key is not a tuple or if argument resolve_tuples is False

    resolve_intervals: bool
        Passed argument resolve_intervals
    """

    def __init__(self, key: Union[int, Tuple[int, int], FeatureDescriptor], resolve_intervals: bool = False):
        if isinstance(key, FeatureDescriptor):
            self._build_from_descriptor(key)
        else:
            self._check_consistency(key)
            self.org_key = key
            self.key = self._resolve_intervals(key, resolve_intervals)
            self.resolve_tuples = resolve_intervals

    def _build_from_descriptor(self, descriptor):
        self.key = descriptor.key
        self.org_key = descriptor.org_key
        self.resolve_tuples = descriptor.resolve_tuples

    def __len__(self):
        return self.comparator[0]

    @cached_property
    def string_rep(self):
        """ Provides a printing representation for the FeatureDescriptor printing interval configurations as number of
        intervals and interval width separated via a forward slash.

        Returns
        -------
            string representation: str
        """
        if isinstance(self.key, int):
            return str(self.key)
        return f'{self.key[0]}/{self.key[1]}'

    @cached_property
    def comparator(self):
        """Provides a feature configuration representation allowing comparison of FeatureDescriptors.
        """
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

    def _resolve_intervals(self, key, resolve_tuple):
        if resolve_tuple and isinstance(key, tuple):
            return key[0] * key[1]  # consistency has already been checked at this point
        return key

    def __hash__(self):
        return self.key.__hash__()

    #
    # Comparator implementations
    #

    def _feature_cast(f):
        @wraps(f)
        def wrapper(s, x):
            if not isinstance(x, FeatureDescriptor):
                x = FeatureDescriptor(x, resolve_intervals=s.resolve_tuples)
            return f(s, x)
        return wrapper

    @_feature_cast
    def __le__(self, x: Union[FeatureDescriptor, int, Tuple[int, int]]):
        """A FeatureDescriptor is less or equal to another FeatureDescriptor, if it selects more features (intervals
        resolved to the number of constituent features) or, in case of equality, if the number of intervals is smaller.
        """
        for i in range(len(self.comparator)):
            if self.comparator[i] < x.comparator[i]:
                return True
            elif self.comparator[i] > x.comparator[i]:
                return False
        return True

    @_feature_cast
    def __ge__(self, x: FeatureDescriptor):
        """A FeatureDescriptor is greater or equal to another FeatureDescriptor, if it selects more features (intervals
        resolved to the number of constituent features) or, in case of equality, if the number of intervals is larger.
        """
        for i in range(len(self.comparator)):
            if self.comparator[i] > x.comparator[i]:
                return True
            elif self.comparator[i] < x.comparator[i]:
                return False
        return True

    #
    # Derived comparison functions
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

    def get_configuration_for(self, selector: SpectralSelector):
        """ Translate and return the feature configuration for a given :class:`~auswahl.SpectralSelector`.

        Parameters
        ----------
        selector: SpectralSelector
            SpectralSelector instance
        """
        if isinstance(selector, PointSelector):
            if self.resolve_tuples:
                return self.key
            else:
                # return the number of overall features to be selected
                return self.key[0] * self.key[1]
        else:
            # return the interval configuration
            return self.key[0], self.key[1]


class SpectralSelector(SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    """ Top level base class for all Auswahl selectors.

    Provides subclassing of all relevant sklearn classes, common cross validationa and hyperparameter optimization functionality.

    Parameters
    ----------
    model_hyperparams: dict
        Dictionary of hyperparameters following the sklearn convention for
        the estimator underlying the selection algorithm.

    n_cv_folds: int
        Number of cross validation runs during model fitting

    random_state: Union[int, np.random.RandomState]
        random state of the selector

    n_jobs: int, default=1
         Number of threads to be used to execute the selection method
    """

    def __init__(self, model_hyperparams: Union[dict, List[dict]], n_cv_folds: int,
                 random_state: Union[int, RandomState] = None, n_jobs: int = 1):
        if model_hyperparams is not None and not isinstance(model_hyperparams, (list, dict)):
            raise ValueError("Keyword argument 'model_hyperparams' is expected to be of type dict or list of dicts")

        if not isinstance(n_cv_folds, int) or n_cv_folds <= 0:
            raise ValueError(f'Keyword argument "n_cv_folds" is expected to be a positive integer. Got {n_cv_folds}')

        self.model_hyperparams = model_hyperparams
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

    def evaluate(self, X, y, model, do_cv=True, *args):
        """Conduct a cross validationand hyperparameter optimization of the underlying estimator model.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Spectral data to be fitted

        y: array-like, shape (n_samples,)
            Regression targets

        model: BaseEstimator
            Regression model

        do_cv: bool, default=True
            If True, the model is fitted to the data and a cross validation score is provided

        *args: arbitrary payload
            Arbitrary payload returned with the evaluation result. Used for instance for
            identification of threads, if multiple models are evaluated in parallel

        Returns
        -------
        tuple: float, BaseEstimator
            cross validation score if requested (otherwise None) and fitted estimator
        """

        model = PLSRegression() if model is None else clone(model)
        model.n_components = min(model.n_components, X.shape[1])
        if self.model_hyperparams is None:  # no hyperparameter optimization; conduct a simple CV
            cv_scores = None
            if do_cv:
                cv_scores = np.mean(cross_val_score(model, X, y, cv=self.n_cv_folds, scoring='neg_mean_squared_error'))
            model.fit(X, y)
            return cv_scores, model, *args
        else:
            cv = GridSearchCV(model, self.model_hyperparams, cv=self.n_cv_folds, scoring='neg_mean_squared_error')
            cv.fit(X, y)
            return cv.best_score_, cv.best_estimator_, *args

    def fit(self, X, y, mask=None):
        """Run the feature selection process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        mask: array-like of shape (n_features,)
            Mask indicating (values == 0), which features are not to be taken into account during the feature selection

        Returns
        -------
        SpectralSelector : self
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

    def get_best_estimator(self) -> sklearn.base.BaseEstimator:
        """Retrieve the best estimator model fitted on the selected features

           Returns
           -------
           best model fitted on selected features: sklearn.base.BaseEstimator
        """
        check_is_fitted(self)
        if not hasattr(self, 'best_model_'):
            raise NotImplementedError("Make sure, that after fit has been called on the selector, the selector "
                                      "provides the optimally configured estimator for the selected features as "
                                      "attribute 'best_model_'")
        return self.best_model_

    def reseed(self, seed: Union[int, RandomState]):
        """ Random state updating interface for benchmarking. Selector methods with more complex internal structure
        (such as methods wrapping other methods) are required to override this function accordingly.
        """
        self.random_state = seed

    def rethread(self, n_jobs: int):
        """ n_jobs updating interface for benchmarking. Selector methods with more complex internal structure
        (such as methods wrapping other methods) are required to override this function accordingly.
        """
        self.n_jobs = n_jobs

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

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
    n_features_to_select : int or float, default=1
        Number of features to select

    model_hyperparams: dict
        Dictionary of estimator hyperparameters following the sklearn convention.

    n_cv_folds: int
        Number of cross validation runs during model fitting

    random_state: Union[int, np.random.RandomState]
        Random state of the selector

    n_jobs: int, default=1
         Number of threads to be used to execute the selection method
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = 1,
                 model_hyperparams: Union[dict, List[dict]] = None,
                 n_cv_folds: int = 2,
                 random_state: Union[int, RandomState] = None,
                 n_jobs: int = 1):
        self.n_features_to_select = n_features_to_select
        super().__init__(model_hyperparams, n_cv_folds, random_state, n_jobs)

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
        self.n_features_to_select = feature_descriptor.get_configuration_for(self)


class IntervalSelector(SpectralSelector, metaclass=ABCMeta):
    """Base class for feature selection methods that select consecutive chunks (intervals) of features.

    Parameters
    ----------
    n_intervals_to_select : int, default=1
        Number of intervals to select.

    interval_width : int or float, default=1
        Number of features that form an interval

    model_hyperparams: dict
        Dictionary of estimator hyperparameters following the sklearn convention.

    n_cv_folds: int
        Number of cross validation runs during model fitting

    random_state: Union[int, np.random.RandomState]
        Random state of the selector

    n_jobs: int, default=1
         Number of threads to be used to execute the selection method
    """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = 1, n_cv_folds: int = 1,
                 model_hyperparams: Union[dict, List[dict]] = None,
                 random_state: Union[int, RandomState] = None,
                 n_jobs: int = 1):
        self.n_intervals_to_select = n_intervals_to_select
        self.interval_width = interval_width
        super().__init__(model_hyperparams, n_cv_folds, random_state, n_jobs)

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
        self.n_intervals_to_select, self.interval_width = feature_descriptor.get_configuration_for(self)


class Convertible(metaclass=ABCMeta):
    """Selectors subclassing :class:`~auswah.PointSelector`, which provide a global score for each feature, can be
    made eligible for a :class:`~auswahl.PointSelector` to :class:`~auswahl.IntervalSelector` conversion facilitated by :class:`~auswahl.PseudoIntervalSelector` by
    inheriting from this class.
    """

    @abstractmethod
    def get_feature_scores(self) -> np.ndarray:
        """Provide scores of all features

           Returns
           -------
           feature scores: np.ndarray of shape [n_features,]
        """
        ...
