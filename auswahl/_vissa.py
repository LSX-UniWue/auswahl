from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state, check_scalar

from ._base import PointSelector


class VISSA(PointSelector):
    """Feature Selection with Variable Iterative Space Shrinkage Approach (VISSA).

    The variable importance is calculated according to  Deng et al. [1]_.

    Read more in the :ref:`User Guide <vip>`.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        Number of features to select.

    n_submodels : int, default=1000
        Number of submodels fitted in each VISSA iteration.

    ratio_submodel_selection : float, default=0.05
        Ratio of submodels that are used to compute the weights for selecting a feature in the next iteration.

    max_iter : int, default=100
        Maximum number of iterations.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

    n_cv_folds : int, default=5
        Number of cross validation folds used in the evaluation of feature sets.

    random_state : int or numpy.random.RandomState, default=None
        Seed for the random subset sampling. Pass an int for reproducible output across function calls.

    n_jobs : int, default=1
        Number of parallel threads to calculate VISSA

    Attributes
    ----------
    frequency_ : ndarray of shape (n_features,)
        VISSA importance scores for variables.

    support_ : ndarray of shape (n_features,)
        Mask of selected features. The highest weighted features are selected

    References
    ----------
    .. [1] Bai-chuan Deng, Yong-huan Yun, Yi-zeng Liang, Lun-shao Yi,
           'A novel variable selection approach that iteratively optimizes variable space using weighted binary
            matrix sampling',
           Analyst, 139, 4836–-4845, 2014.

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import VISSA
    >>> np.random.seed(1337)
    >>> X = np.random.randn(100, 10)
    >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
    >>> selector = VISSA(n_features_to_select=2, n_submodels=100)
    >>> selector.fit(X, y).get_support()
    array([ True, False, False, False, False, True, False, False, False, False])
    """

    def __init__(self,
                 n_features_to_select: int = None,
                 n_submodels: int = 1000,
                 ratio_submodel_selection: float = 0.05,
                 max_iter: int = 100,
                 pls: PLSRegression = None,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 n_cv_folds: int = 5,
                 random_state: Union[int, np.random.RandomState] = None,
                 n_jobs: int = 1):
        super().__init__(n_features_to_select,
                         model_hyperparams=model_hyperparams,
                         n_cv_folds=n_cv_folds,
                         random_state=random_state,
                         n_jobs=n_jobs)
        self.pls = pls
        self.n_submodels = n_submodels
        self.ratio_submodel_selection = ratio_submodel_selection
        self.max_iter = max_iter

    def _fit_model_on_subset(self, X, y, mask, model):
        if mask.sum() < 1:
            return -np.inf
        else:
            return self.evaluate(X[:, mask], y, model)[0]

    def _fit(self, X, y, n_features_to_select):
        self._check_n_submodels()
        self._check_ratio_submodel_selection()
        self._check_max_iter()
        random_state = check_random_state(self.random_state)

        n_best_models = int(self.ratio_submodel_selection * self.n_submodels)
        n_best_models = np.clip(n_best_models, 2, self.n_submodels)
        n_features = X.shape[1]

        selection_frequency = [self.n_submodels // 2] * n_features

        with Parallel(n_jobs=self.n_jobs) as parallel:
            last_best_score = -np.inf
            for i in range(self.max_iter):
                sampling_mask = np.stack([random_state.permutation([True] * p + [False] * (self.n_submodels - p))
                                          for p in selection_frequency], axis=1)

                scores = parallel(delayed(self._fit_model_on_subset)(X, y, mask, self.pls)
                                  for mask in sampling_mask)
                scores = np.array(scores)

                best_models = np.argsort(scores)[-n_best_models:]
                new_frequency = (sampling_mask[best_models].mean(axis=0) * self.n_submodels).astype(int)
                best_score = np.mean(scores[best_models])

                if np.isclose(last_best_score, best_score) or (np.sum(new_frequency > 0) < n_features_to_select):
                    break
                last_best_score = best_score
                selection_frequency = new_frequency

        self.n_iter_ = i
        self.frequency_ = selection_frequency
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[np.argsort(selection_frequency)[-n_features_to_select:]] = 1

        _, self.best_model_ = self.evaluate(X[:, self.support_], y, self.pls, do_cv=False)
        return self

    def _check_n_submodels(self):
        check_scalar(self.n_submodels, 'n_submodels', target_type=int, min_val=2)

    def _check_ratio_submodel_selection(self):
        check_scalar(self.ratio_submodel_selection,
                     name='ratio_submodel_selection',
                     target_type=float,
                     min_val=0,
                     max_val=1,
                     include_boundaries='right')

    def _check_max_iter(self):
        check_scalar(self.max_iter, 'max_iter', target_type=int, min_val=1)
