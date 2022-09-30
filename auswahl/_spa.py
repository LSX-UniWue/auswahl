from typing import Union, List, Dict

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from ._base import PointSelector


class SPA(PointSelector):
    """Feature selection with the Successive Projection Algorithm (SPA).

    The Successive Projections Algorithm conducts feature selection according to Araújo et al. [1]_.
    The algorithm aims to find a set of features exhibiting minimal collinearity.

    Read more in the :ref:`User Guide <spa>`.

    Parameters
    ----------
    n_features_to_select : int, default=None
        Upper bound of features to select.

    n_cv_folds : int, default=5
        Number of cross validation folds used in the evaluation of feature sets.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

    n_jobs : int, default=1
        Number of jobs used for parallel calculation of SPA

    Attributes
    ----------
    support_ : ndarray fo shape (n_features,)
        Mask of selected features

    References
    ----------
    .. [1] Mário César Ugulino Araújo,Teresa Cristina Bezerra Saldanha, Roberto Kawakami Harrop Galvao,
           Takashi Yoneyama, Henrique Caldas Chame and Valeria Visani,
           The successive projections algorithm for variable selection in spectroscopic multicomponent analysis,
           Chemometrics and Intelligent Laboratory Systems, 57, 65-73, 2001

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import SPA
    >>> np.random.seed(1337)
    >>> X = np.random.randn(1000, 10)
    >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
    >>> selector = SPA(n_features_to_select=2)
    >>> selector.fit(X, y).get_support()
    array([ True, False, False, False, False, True, False, False, False, False])
    """
    
    def __init__(self, 
                 n_features_to_select: int = None,
                 n_cv_folds: int = 5,
                 pls: PLSRegression = None,
                 n_jobs: int = 1,
                 model_hyperparams: Union[Dict, List[Dict]] = None):
        
        super().__init__(n_features_to_select, model_hyperparams, n_cv_folds, n_jobs=n_jobs)
        
        self.pls = pls

    def _fit_spa(self, X, y, n_features_to_select, pls, seed):

        wavelengths = [seed]
        current = X[:, seed:seed + 1]
        rest = np.delete(X, seed, 1)

        wavelength_map = np.arange(X.shape[1])
        wavelength_map = np.delete(wavelength_map, seed)

        for j in range(n_features_to_select - 1):
            current = current / np.linalg.norm(current, ord=2)
            projections = rest - current @ np.transpose(np.transpose(rest) @ current)
            projection_distances = np.linalg.norm(projections, ord=2, axis=0)

            next_index = np.argmax(projection_distances)
            current = projections[:, next_index:next_index + 1]
            rest = np.delete(projections, next_index, 1)

            wavelengths.append(wavelength_map[next_index])
            wavelength_map = np.delete(wavelength_map, next_index)

        score, model = self.evaluate(X[:, wavelengths], y, self.pls)
        return score, model, wavelengths

    def _fit(self, X, y, n_features_to_select):
        candidates = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_spa)(X,
                                                  y,
                                                  n_features_to_select,
                                                  self.pls,
                                                  i) for i in range(X.shape[-1]))
        score, model, opt_set = max(candidates, key=lambda x: x[0])
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[opt_set] = True
        self.best_model_ = model

