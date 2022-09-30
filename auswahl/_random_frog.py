from abc import ABCMeta, abstractmethod
from typing import Union, Dict, List
from warnings import warn

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted, check_random_state, check_scalar

from ._base import PointSelector, IntervalSelector, SpectralSelector
from .util import get_coef_from_pls


class _RandomFrog(SpectralSelector, BaseEstimator, metaclass=ABCMeta):
    """Mixin for the Random Frog feature selection method.

    The py:class:`auswahl.RandomFrogPointSelector` can be used for Wavelength Point selection and the
    py:class:`auswahl.RandomFrogIntervalSelector` can be used for Wavelength Interval selection.
    """

    def _select(self, X, y,
                n_features: int,
                n_features_to_select: Union[int, float] = None,
                n_iterations: int = 10000,
                n_initial_features: Union[int, float] = 0.1,
                variance_factor: float = 0.3,
                subset_expansion_factor: float = 3,
                acceptance_factor: float = 0.1,
                pls: PLSRegression = None,
                random_state: Union[int, np.random.RandomState] = None):
        # Perform parameter checks
        self._check_n_iterations(n_iterations)
        self._check_subset_expansion_factor(subset_expansion_factor)
        self._check_acceptance_factor(acceptance_factor)

        n_initial_features = self._check_n_initial_features(X, n_initial_features)
        variance_factor = self._check_variance_factor(variance_factor)
        random_state = check_random_state(random_state)

        # Initialize estimator, feature sets and frequency counter
        pls = PLSRegression() if pls is None else clone(pls)
        n_components = pls.n_components

        all_features = np.arange(n_features)
        selected_features = random_state.choice(n_features, n_initial_features, replace=False)
        self.frequencies_ = np.zeros(n_features)

        # Random Frog Iteration
        for i in range(n_iterations):
            n_selected_features = len(selected_features)
            n_candidate_features = random_state.normal(n_selected_features, n_selected_features * variance_factor)
            n_candidate_features = np.clip(round(n_candidate_features), n_components, n_features)

            if n_candidate_features < n_selected_features:
                # Reduction step
                features_to_explore = selected_features
            elif n_candidate_features > n_selected_features:
                # Expansion step
                n_diff = n_candidate_features - n_selected_features
                non_selected_features = np.setdiff1d(all_features, selected_features)
                n_features_to_explore = min(subset_expansion_factor * n_diff, len(non_selected_features))
                additional_features = random_state.choice(non_selected_features, n_features_to_explore, replace=False)
                features_to_explore = np.union1d(selected_features, additional_features)
            else:
                # Skip step
                self.frequencies_[selected_features] += 1
                continue

            # Determine the candidate feature selection
            _, pls = self.evaluate(X[:, self._idx_to_mask(features_to_explore)], y, pls)
            absolute_coefficients = self._get_feature_score_from_model(pls, features_to_explore)
            selection_idx = np.argsort(absolute_coefficients)[-n_candidate_features:]
            candidate_features = features_to_explore[selection_idx]

            # Score the current feature selection and the candidate feature selection
            pls.n_components = min(pls.n_components, len(selected_features))
            selected_features_score, _ = self.evaluate(X=X[:, self._idx_to_mask(selected_features)], y=y, model=pls)

            pls.n_components = min(pls.n_components, len(candidate_features))
            candidate_features_score, _ = self.evaluate(X=X[:, self._idx_to_mask(candidate_features)], y=y, model=pls)

            # Update the feature selection
            if candidate_features_score >= selected_features_score:
                selected_features = candidate_features
            elif random_state.random() < acceptance_factor * (selected_features_score / candidate_features_score):
                selected_features = candidate_features
            self.frequencies_[selected_features] += 1

        self.support_ = self._generate_mask_from_frequencies(n_features_to_select)
        self.best_model_ = pls.fit(self.transform(X), y)

        return self

    @abstractmethod
    def _idx_to_mask(self, feature_idx):
        pass

    @abstractmethod
    def _generate_mask_from_frequencies(self, n_features_to_select):
        pass

    @abstractmethod
    def _get_feature_score_from_model(self, pls, feature_idx):
        pass

    @staticmethod
    def _check_n_iterations(n_iterations):
        check_scalar(n_iterations, name='n_iterations', target_type=int, min_val=1)

    @staticmethod
    def _check_n_initial_features(X, n_initial_features):
        n_features = X.shape[1]

        if n_initial_features is None:
            n_initial_features = 0.1
        if 0 < n_initial_features < 1:
            n_initial_features = max(1, int(n_initial_features * n_features))

        if (n_initial_features < 1) or (n_initial_features > n_features):
            raise ValueError('n_initial_features has to be either an int in {1, ..., n_features}'
                             f'or a float in (0, 1); got {n_initial_features}')

        return n_initial_features

    @staticmethod
    def _check_variance_factor(variance_factor):
        if variance_factor <= 0:
            warn('variance_factor is negative! abs(variance_factor) will be used for the fit method.')
            variance_factor = abs(variance_factor)
        return variance_factor

    @staticmethod
    def _check_subset_expansion_factor(subset_expansion_factor):
        check_scalar(subset_expansion_factor, name='subset_expansion_factor', target_type=(int, float), min_val=1)

    @staticmethod
    def _check_acceptance_factor(acceptance_factor):
        check_scalar(acceptance_factor, name='acceptance_factor', target_type=(int, float), min_val=0, max_val=1)


class RandomFrog(PointSelector, _RandomFrog):
    r"""Feature selection with the Random Frog method.

    The selection frequencies are computed according to Li et al. [1]_.

    Read more in the :ref:`User Guide <rf>`.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        Number of features to select.

    n_iterations : int, default=10000
        Number of variable selection iterations.
        This variable is called N in the original publication.

    n_initial_features : int or float, default=0.1
        Number of features in the initial  feature subset. If `None`, 10 % of the features are used. If integer, the
        parameter is the size of the initial subset. If float between 0 and 1, it is the fraction of features to use.
        This variable is called Q in the original publication.

    variance_factor : float, default=0.3
        Variance of the normal distribution which samples determine the amount of features that are added or removed
        to the candidate set in each iteration.
        This variable is called θ in the original publication.

    subset_expansion_factor : float, default=3
        Multiple of the number of features that are explored if the candidate subset is expanded. If the current feature
        subset is n and m new features have to be added to the new feature subset, m*subset_expansion_factors features
        are added to a candidate set. After fitting a PLS model, only the n+m features with the highest coefficients are
        kept.
        This variable is called ω in the original publication.

    acceptance_factor : float, default=0.1
        The factor is used to calculate the probability that a feature subset is selected even though it leads to a
        worse cross-validation performance of a fitted PLS model. The probability is computed by multiplying the
        acceptance_factor with the relative decrease of the cross-validated performance score.
        This variable is called η in the original publication.

    n_cv_folds : int, default=5
        Number of cross validation folds used to evaluate the features.

    n_jobs : int, default=1
        Number of parallel processes used to fit the PLS models on the cross-validation splits.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

    random_state : int or numpy.random.RandomState, default=None
        Seed for the random subset sampling. Pass an int for reproducible output across function calls.

    Attributes
    ----------
    frequencies_ : ndarray of shape (n_features,)
        Number of times each feature has been selected after all iterations.

    support_ : ndarray of shape (n_features,)
        Mask of selected features.

    References
    ----------
    .. [1] Hong-Dong Li and Qing-Song Xu and Yi-Zeng Liang,
           'Random frog: An efficient reversible jump Markov Chain Monte Carlo-like approach for variable selection
           with applications to gene selection and disease classification',
           Analytica Chimica Acta, 740, 20-26, 2012

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import RandomFrog
    >>> np.random.seed(1337)
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
    >>> selector = RandomFrog(n_features_to_select=2, n_iterations=1000)
    >>> selector.fit(X, y).get_support()
    array([ True, False, False, False, False, True, False, False, False, False])
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 n_iterations: int = 10000,
                 n_initial_features: Union[int, float] = 0.1,
                 variance_factor: float = 0.3,
                 subset_expansion_factor: float = 3,
                 acceptance_factor: float = 0.1,
                 pls: PLSRegression = None,
                 n_cv_folds: int = 5,
                 n_jobs: int = 1,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 random_state: Union[int, np.random.RandomState] = None):
        super().__init__(n_features_to_select, model_hyperparams=model_hyperparams,
                         random_state=random_state, n_jobs=n_jobs)
        self.n_iterations = n_iterations
        self.n_initial_features = n_initial_features
        self.variance_factor = variance_factor
        self.subset_expansion_factor = subset_expansion_factor
        self.acceptance_factor = acceptance_factor
        self.n_cv_folds = n_cv_folds
        self.pls = pls

    def _fit(self, X, y, n_features_to_select):
        self.n_features_ = X.shape[1]
        return super()._select(X, y,
                               n_features=self.n_features_,
                               n_features_to_select=n_features_to_select,
                               n_iterations=self.n_iterations,
                               n_initial_features=self.n_initial_features,
                               variance_factor=self.variance_factor,
                               subset_expansion_factor=self.subset_expansion_factor,
                               acceptance_factor=self.acceptance_factor,
                               pls=self.pls,
                               random_state=self.random_state)

    def _idx_to_mask(self, feature_idx):
        mask = np.zeros(self.n_features_, dtype=bool)
        mask[feature_idx] = 1
        return mask

    def _generate_mask_from_frequencies(self, n_features_to_select):
        mask = np.zeros(len(self.frequencies_), dtype=bool)
        selected_idx = np.argsort(self.frequencies_)[-n_features_to_select:]
        mask[selected_idx] = 1
        return mask

    def _get_feature_score_from_model(self, pls, feature_idx):
        return abs(get_coef_from_pls(pls).squeeze())


class IntervalRandomFrog(IntervalSelector, _RandomFrog):
    r"""Feature selection with the Interval Random Frog (iRF) method.

    The selection frequencies are computed according to Yun et al. [1]_.

    Read more in the :ref:`User Guide <irf>`.

    Parameters
    ----------
    n_intervals_to_select : int, default=1
        Number of intervals to select.

    interval_width: int or float, default=None
        Size of the selected intervals. If `None`, the intervals are n_features/2 long. If integer, the parameter
        directly defines the number of consecutive features that form an interval. If float between 0 and 1, the
        intervals are n_features*n_intervals_to_select long.

    n_iterations : int, default=10000
        Number of variable selection iterations.
        This variable is called N in the original publication.

    n_initial_intervals : int or float, default=0.1
        Number of intervals in the initial  interval subset. If `None`, 10 % of the intervals are used. If integer, the
        parameter is the size of the initial subset. If float between 0 and 1, it is the fraction of intervals to use.
        This variable is called Q in the original publication.

    variance_factor : float, default=0.3
        Variance of the normal distribution which samples determine the amount of intervals that are added or removed
        to the candidate set in each iteration.
        This variable is called θ in the original publication.

    subset_expansion_factor : float, default=3
        Multiple of the number of intervals that are explored if the candidate subset is expanded. If the current
        interval subset is n and m new intervals have to be added to the new interval subset, m*subset_expansion_factors
        intervals are added to a candidate set. After fitting a PLS model, only the n+m intervals with the highest
        coefficients are kept.
        This variable is called ω in the original publication.

    acceptance_factor : float, default=0.1
        The factor is used to calculate the probability that an interval subset is selected even though it leads to a
        worse cross-validation performance of a fitted PLS model. The probability is computed by multiplying the
        acceptance_factor with the relative decrease of the cross-validated performance score.
        This variable is called η in the original publication.

    n_cv_folds : int, default=5
        Number of cross validation folds used to evaluate the features.

    n_jobs : int, default=1
        Number of parallel processes used to fit the PLS models on the cross-validation splits.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

    random_state : int or numpy.random.RandomState, default=None
        Seed for the random subset sampling. Pass an int for reproducible output across function calls.

    Attributes
    ----------
    frequencies_ : ndarray of shape (n_features,)
        Number of times each interval has been selected after all iterations.

    support_ : ndarray of shape (n_features,)
        Mask of selected intervals.

    References
    ----------
    .. [1] Yong-Huan Yun and Hong-Dong Li and Leslie R. E. Wood and Wei Fan and Jia-Jun Wang and Dong-Sheng Cao and
           Qing-Song Xu and Yi-Zeng Liang,
           'An efficient method of wavelength interval selection based on random frog for multivariate spectral
           calibration',
           Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 111, 31-36, 2013

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import IntervalRandomFrog
    >>> np.random.seed(1337)
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 5] - 3 * X[:, 6]  # y only depends on two intervals
    >>> selector = IntervalRandomFrog(n_intervals_to_select=2, interval_width=2, n_iterations=1000, random_state=7331)
    >>> selector.fit(X, y).get_support()
    array([ True, True, False, False, False, True, True, False, False, False])
    """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None,
                 n_iterations: int = 10000,
                 n_initial_intervals: Union[int, float] = 0.1,
                 variance_factor: float = 0.3,
                 subset_expansion_factor: float = 3,
                 acceptance_factor: float = 0.1,
                 n_cv_folds: int = 5,
                 n_jobs: int = 1,
                 pls: PLSRegression = None,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 random_state: Union[int, np.random.RandomState] = None):
        super().__init__(n_intervals_to_select, interval_width,
                         model_hyperparams=model_hyperparams, n_cv_folds=n_cv_folds,
                         random_state=random_state, n_jobs=n_jobs)
        self.n_iterations = n_iterations
        self.n_initial_intervals = n_initial_intervals
        self.variance_factor = variance_factor
        self.subset_expansion_factor = subset_expansion_factor
        self.acceptance_factor = acceptance_factor
        self.n_cv_folds = n_cv_folds
        self.pls = pls

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        self.n_windows_ = X.shape[1] - interval_width + 1
        self.interval_width_ = self._check_interval_width(X)
        return super()._select(X, y,
                               n_features=self.n_windows_,
                               n_features_to_select=n_intervals_to_select,
                               n_iterations=self.n_iterations,
                               n_initial_features=self.n_initial_intervals,
                               variance_factor=self.variance_factor,
                               subset_expansion_factor=self.subset_expansion_factor,
                               acceptance_factor=self.acceptance_factor,
                               pls=self.pls,
                               random_state=self.random_state)

    def _idx_to_mask(self, feature_idx):
        mask = np.zeros(self.n_windows_ + self.interval_width_ - 1, dtype=bool)
        for idx in feature_idx:
            mask[idx:idx + self.interval_width_] = 1
        return mask

    def _generate_mask_from_frequencies(self, n_features_to_select):
        mask = np.zeros(len(self.frequencies_) + self.interval_width_ - 1, dtype=bool)
        scores = self.frequencies_.copy()
        for i in range(n_features_to_select):
            best_idx = np.argmax(scores)
            mask[best_idx:best_idx + self.interval_width_] = True
            start = np.clip(best_idx - self.interval_width_ + 1, 0, np.inf).astype(int)
            end = best_idx + self.interval_width_
            scores[start:end] = -1
        return mask

    def _get_feature_score_from_model(self, pls, feature_idx):
        scores = np.zeros(self.n_windows_ + self.interval_width_ - 1)
        scores[self._idx_to_mask(feature_idx)] = abs(get_coef_from_pls(pls).squeeze())
        scores = [sum(scores[idx:idx + self.interval_width_]) for idx in feature_idx]
        return scores
