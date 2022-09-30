import warnings
from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .util import get_coef_from_pls
from ._base import PointSelector


class CARS(PointSelector):
    """Feature selection with Competitive Adaptive Reweighted Sampling (CARS).

    The feature selection is conducted according to Li et al. [1]_.
    Since CARS is not designed to return a feature set of a specifc size, the implementation at hand is an adaption of
    the algorithm of Li et al. for this specific setting.

    Read more in the :ref:`User Guide <cars>`.

    Parameters
    ----------
    n_features_to_select : int, default=None
        Upper bound of features to select.

    n_cars_runs : int, default=20
        Number of individual CARS runs to estimate the selection stability of wavelengths

    n_jobs : int, default=2
        Number of parallel workers

    n_sample_runs : int, default=100
        Number of sampling runs.

    fit_samples_ratio : float, default=0.9
        Ratio of samples used to fit the regression model, used for scoring of features.

    n_cv_folds : int, default=5
        Number of cross validation folds used in the evaluation of feature sets.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class.
        Use this to adjust the hyperparameters of the PLS method.

    random_state : int or numpy.random.RandomState, default=None
        Seed for the random subset sampling. Pass an int for reproducible output across function calls.

    Attributes
    ----------
    support_ : ndarray of shape (n_features,)
        Mask of selected features

    References
    ----------
    .. [1] Hongdong Li,Yizeng Liang, Qingsong Xu and Dongsheng Cao,
           Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration,
           Analytica Chimica Acta, 648, 77-84, 2009

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import CARS
    >>> X = np.random.randn(100, 15)
    >>> y = 5 * X[:, -2] - 2 * X[:, -1]  # y only depends on two features
    >>> selector = CARS(n_features_to_select=2,n_sample_runs = 100)
    >>> selector.fit(X, y).get_support()
    array([False, False, False, False, False, False, False, False, False, False, False, False, False, True, True])
    """

    def __init__(self,
                 n_features_to_select: int = None,
                 n_cars_runs: int = 20,
                 n_jobs: int = 1,
                 n_sample_runs: int = 100,
                 fit_samples_ratio: float = 0.9,
                 n_cv_folds: int = 5,
                 pls: PLSRegression = None,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 random_state: Union[int, np.random.RandomState] = None):

        super().__init__(n_features_to_select, model_hyperparams, n_cv_folds, random_state, n_jobs)

        self.pls = pls
        self.n_cars_runs = n_cars_runs
        self.n_sample_runs = n_sample_runs
        self.fit_samples_ratio = fit_samples_ratio

    def _prepare_edf_schedule(self, n_wavelengths, ):
        a = (n_wavelengths/2)**(1/(self.n_sample_runs-1))
        k = np.log(n_wavelengths/2) / (self.n_sample_runs-1)

        iterations = -k * (np.arange(0, self.n_sample_runs) + 1)
        selection_ratios = a * np.exp(iterations)

        return (selection_ratios*n_wavelengths + 1e-10).astype('int')

    def _get_wavelength_weights(self, X, y, n_fit_samples, wavelengths, pls, random_state):
        fitting_samples = random_state.choice(X.shape[0],
                                              n_fit_samples,
                                              replace=False)

        x_pls_fit = X[fitting_samples, :][:, wavelengths]
        y_pls_fit = y[fitting_samples]

        _, model = self.evaluate(x_pls_fit, y_pls_fit, pls, do_cv=False)
        weights = np.abs(get_coef_from_pls(model)).flatten()
        wavelength_weights = np.zeros(X.shape[1])
        wavelength_weights[wavelengths] = weights

        return wavelength_weights

    def _fit_cars(self, X, y, n_features_to_select, edf_schedule, pls, seed):
        pls = PLSRegression() if pls is None else clone(pls)
        random_state = check_random_state(seed)

        n_fit_samples = int(X.shape[0] * self.fit_samples_ratio)

        wavelengths = np.arange(X.shape[1])
        for i in range(self.n_sample_runs):

            weights = self._get_wavelength_weights(X, y,
                                                   n_fit_samples, wavelengths,
                                                   pls, random_state)
            # ensure, that at least n_features_to_select scheduled
            scheduled = max(edf_schedule[i], n_features_to_select)
            wavelengths = np.argsort(-weights)[:scheduled]
            wavelength_probs = weights[wavelengths] / np.sum(weights[wavelengths])

            # ensure, that n_features_to_select features are always selected
            base_wavelengths = random_state.choice(wavelengths,
                                                   n_features_to_select,
                                                   replace=False,
                                                   p=wavelength_probs)

            additional_wavelengths = random_state.choice(wavelengths,
                                                         scheduled - n_features_to_select,
                                                         replace=True,
                                                         p=wavelength_probs)

            wavelengths = np.concatenate([base_wavelengths, additional_wavelengths])
            wavelengths = np.unique(wavelengths)

            if wavelengths.shape[0] == n_features_to_select:
                break

        score, model = self.evaluate(X[:, wavelengths], y, pls)
        return score, wavelengths, model

    def _calculate_feature_importance(self, n_features, selection_candidates):
        importance = np.zeros((n_features,))
        for score, wavelengths, _ in selection_candidates:
            importance[wavelengths] = importance[wavelengths] + np.ones((len(wavelengths, )))
        return importance / self.n_cars_runs

    def _fit(self, X, y, n_features_to_select):
        self._check_n_sample_runs()
        self._check_fit_samples_ratio()

        random_state = check_random_state(self.random_state)
        seeds = random_state.random_integers(0, 1000000, self.n_cars_runs)

        edf_schedule = self._prepare_edf_schedule(X.shape[1])

        candidates = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_cars)(X,
                                                                          y,
                                                                          n_features_to_select,
                                                                          edf_schedule,
                                                                          self.pls,
                                                                          seeds[i]) for i in range(self.n_cars_runs))
        score, opt_wavelengths, best_model = max(candidates, key=lambda x: x[0])
        self.feature_importance_ = self._calculate_feature_importance(X.shape[1], candidates)
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[opt_wavelengths] = True
        self.best_model_ = best_model

    def _check_fit_samples_ratio(self):
        if self.fit_samples_ratio < 0:
            raise ValueError('fit_sample_ratio is required to be in [0,1]. ' 
                             f'Got {self.fit_samples_ratio}')
        if self.fit_samples_ratio > 1:
            warnings.warn(f'fit_samples_ratio clipped to 1. Got {self.fit_samples_ratio}')
            self.fit_samples_ratio = 1

    def _check_n_sample_runs(self):
        if self.n_sample_runs < 2:
            raise ValueError('n_sample_runs is required to be >= 2. '
                             f'Got {self.n_sample_runs}')
