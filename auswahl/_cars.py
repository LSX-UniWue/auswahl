from typing import Union, Dict

import numpy as np
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

import warnings

from auswahl._base import PointSelector


class CARS(PointSelector):

    """Feature selection with Competitive Adaptive Reweighted Sampling (CARS).
    
    
        The feature selection is conducted according to Li et al. [1]_.
        Since CARS is not designed to return a feature set of a specifc size, the implementation at hand
        is an adaption of the algorithm of Li et al. for this specific setting.
    
        Parameters
        ----------
        
        n_features_to_select : int, default=None
            Upper bound of features to select.

        n_cars_runs : int, default=20
            Number of individual CARS rus to estimate the selection stability of wavelengths

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

        selection_count_ : ndarray of shape (n_features,)
            Indicating the number of times a wavelength has been selected during CARS sample runs
        
    
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
        >>> selector.fit(X, y)
        >>> selector.get_support()
        array([False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True])
    
    """
    
    def __init__(self, 
                 n_features_to_select: int = None,
                 n_cars_runs: int = 20,
                 n_jobs: int = 1,
                 n_sample_runs: int = 100,
                 fit_samples_ratio: float = 0.9,
                 n_cv_folds: int = 5,
                 pls: PLSRegression = None,
                 random_state: Union[int, np.random.RandomState] = None):
        
        super().__init__(n_features_to_select)
        
        self.pls = pls
        self.n_jobs = n_jobs
        self.n_cars_runs = n_cars_runs
        self.n_sample_runs = n_sample_runs
        self.fit_samples_ratio = fit_samples_ratio
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        
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
        
        pls.fit(x_pls_fit, y_pls_fit)
        weights = np.abs(pls.coef_).flatten()
        wavelength_weights = np.zeros(X.shape[1])
        wavelength_weights[wavelengths] = weights
        
        return wavelength_weights
    
    def _evaluate(self, X, y, wavelengths, pls):
        
        cv_scores = cross_val_score(pls,
                                    X[:, wavelengths],
                                    y, 
                                    cv=self.n_cv_folds, 
                                    scoring='neg_mean_squared_error')
        return np.mean(cv_scores)
        
    def _fit_cars(self, X, y, n_features_to_select, pls):
        
        pls = PLSRegression() if pls is None else clone(pls)
        random_state = check_random_state(self.random_state)
        
        self._check_n_sample_runs()
        self._check_fit_samples_ratio()
        
        edf_schedule = self._prepare_edf_schedule(X.shape[1])
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
        
        return wavelengths

    def _fit(self, X, y, n_features_to_select):
        candidates = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_cars)(X,
                                                                          y,
                                                                          n_features_to_select,
                                                                          self.pls) for i in range(self.n_cars_runs))
        selection_counter = np.zeros(X.shape[1])
        for prop in candidates:
            selection_counter[prop] = selection_counter[prop] + 1

        self.selection_count_ = selection_counter
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[np.argsort(-selection_counter)[:n_features_to_select]] = True

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_ 
    
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
