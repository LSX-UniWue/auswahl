from typing import Union
from warnings import warn

import numpy as np
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils.validation import check_is_fitted, check_random_state, check_scalar

from auswahl._base import PointSelector


class RandomFrog(PointSelector):
    r"""Feature selection with the Random Frog method.

    The selection frequencies are computed according to Li et al. [1]_.

    Parameters
    ----------
    n_features_to_select: int or float, default=None
        Number of features to select.

    n_iterations: int, default=10000
        Number of variable selection iterations.
        This variable is called N in the original publication.

    n_initial_features: int or float, default=0.1
        Number of features in the initial  feature subset. If `None`, 10 % of the features are used. If integer, the
        parameter is the size of the initial subset. If float between 0 and 1, it is the fraction of features to use.
        This variable is called Q in the original publication.

    variance_factor: float, default=0.3
        Variance of the normal distribution which samples determine the amount of features that are added or removed
        to the candidate set in each iteration.
        This variable is called θ in the original publication.

    subset_expansion_factor: float, default=3
        Multiple of the number of features that are explored if the candidate subset is expanded. If the current feature
        subset is n and m new features have to be added to the new feature subset, m*subset_expansion_factors features
        are added to a candidate set. After fitting a PLS model, only the n+m features with the highest coefficients are
        kept.
        This variable is called ω in the original publication.

    acceptance_factor: float, default=0.1
        The factor is used to calculate the probability that a feature subset is selected even though it leads to a
        worse cross-validation performance of a fitted PLS model. The probability is computed by multiplying the
        acceptance_factor with the relative decrease of the cross-validated performance score.
        This variable is called η in the original publication.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class.
        Use this to adjust the hyperparameters of the PLS method.

    random_state : int or numpy.random.RandomState, default=None
        Seed for the random subset sampling. Pass an int for reproducible output across function calls.

    Attributes
    ----------
    frequencies_: ndarray of shape (n_features,)
        Number of times each feature has been selected after all iterations
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
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
    >>> selector = RandomFrog(n_features_to_select=2)
    >>> selector.fit(X, y)
    >>> selector.get_support()
    array([True, False, False, False, False, True, False, False, False, False])
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 n_iterations: int = 10000,
                 n_initial_features: Union[int, float] = 0.1,
                 variance_factor: float = 0.3,
                 subset_expansion_factor: float = 3,
                 acceptance_factor: float = 0.1,
                 pls: PLSRegression = None,
                 random_state: Union[int, np.random.RandomState] = None):
        super().__init__(n_features_to_select)
        self.n_iterations = n_iterations
        self.n_initial_features = n_initial_features
        self.variance_factor = variance_factor
        self.subset_expansion_factor = subset_expansion_factor
        self.acceptance_factor = acceptance_factor
        self.pls = pls
        self.random_state = random_state

    def _fit(self, X, y, n_features_to_select):
        n_features = X.shape[1]

        # Perform parameter checks
        self._check_n_iterations()
        self._check_subset_expansion_factor()
        self._check_acceptance_factor()

        n_initial_features = self._check_n_initial_features(X)
        variance_factor = self._check_variance_factor()
        random_state = check_random_state(self.random_state)

        # Initialize estimator, feature sets and frequency counter
        pls = PLSRegression() if self.pls is None else clone(self.pls)
        all_features = np.arange(n_features)
        selected_features = random_state.choice(n_features, n_initial_features, replace=False)
        self.frequencies_ = np.zeros(n_features)

        # Random Frog Iteration
        for i in range(self.n_iterations):
            n_selected_features = len(selected_features)
            n_candidate_features = random_state.normal(n_selected_features, n_selected_features * variance_factor)
            n_candidate_features = np.clip(round(n_candidate_features), pls.n_components, n_features)

            if n_candidate_features < n_selected_features:
                # Reduction step
                features_to_explore = selected_features
            elif n_candidate_features > n_selected_features:
                # Expansion step
                n_diff = n_candidate_features - n_selected_features
                non_selected_features = np.setdiff1d(all_features, selected_features)
                n_features_to_explore = min(self.subset_expansion_factor * n_diff, len(non_selected_features))
                additional_features = random_state.choice(non_selected_features, n_features_to_explore, replace=False)
                features_to_explore = np.union1d(selected_features, additional_features)
            else:
                # Skip step
                self.frequencies_[selected_features] += 1
                continue

            pls.fit(X[:, features_to_explore], y)
            absolute_coefficients = abs(pls.coef_.squeeze())
            selection_idx = np.argsort(absolute_coefficients)[-n_candidate_features:]
            candidate_features = features_to_explore[selection_idx]

            cv_split = KFold(n_splits=5, shuffle=False)
            selected_features_score = cross_val_score(pls, X[:, selected_features], y,
                                                      scoring='neg_root_mean_squared_error', cv=cv_split).mean()
            candidate_features_score = cross_val_score(pls, X[:, candidate_features], y,
                                                       scoring='neg_root_mean_squared_error', cv=cv_split).mean()

            if candidate_features_score >= selected_features_score:
                selected_features = candidate_features
            elif random_state.random() < self.acceptance_factor * (selected_features_score / candidate_features_score):
                selected_features = candidate_features
            self.frequencies_[selected_features] += 1

        selected_idx = np.argsort(self.frequencies_)[-n_features_to_select:]
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected_idx] = 1

        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def _check_n_iterations(self):
        check_scalar(self.n_iterations, name='n_iterations', target_type=int, min_val=1)

    def _check_n_initial_features(self, X):
        n_features = X.shape[1]
        n_initial_features = self.n_initial_features

        if n_initial_features is None:
            n_initial_features = 0.1
        if 0 < n_initial_features < 1:
            n_initial_features = max(1, int(n_initial_features * n_features))

        if (n_initial_features < 1) or (n_initial_features > n_features):
            raise ValueError('n_initial_features has to be either an int in {1, ..., n_features}'
                             f'or a float in (0, 1); got {self.n_initial_features}')

        return n_initial_features

    def _check_variance_factor(self):
        variance_factor = self.variance_factor
        if variance_factor <= 0:
            warn('variance_factor is negative! abs(variance_factor) will be used during for the fit method.')
            variance_factor = abs(variance_factor)
        return variance_factor

    def _check_subset_expansion_factor(self):
        check_scalar(self.subset_expansion_factor, name='subset_expansion_factor', target_type=(int, float), min_val=1)

    def _check_acceptance_factor(self):
        check_scalar(self.acceptance_factor, name='acceptance_factor', target_type=(int, float), min_val=0, max_val=1)
