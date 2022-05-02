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


class VISSA(PointSelector):

    """
        Feature Selection with Variable Iterative Space Shrinkage Approach (VISSA).

        The variable importance is calculated according to  Deng et al. [1]_.

        Read more in the :ref:`User Guide <vip>`.

        Parameters
        ----------
        n_features_to_select : int or float, default=None
            Number of features to select.

        n_jobs : int, default=1
            Number of parallel threads to calculate VISSA

        n_submodels : int, default=1000
            Number of submodels fitted in each VISSA iteration

        n_cv_folds : int, default=5
            Number of cross validation folds used in the evaluation of feature sets.

        pls : PLSRegression, default=None
            Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class.
            Use this to adjust the hyperparameters of the PLS method.


        Attributes
        ----------
        weights_ : ndarray of shape (n_features,)
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
        >>> X = np.random.randn(100, 10)
        >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
        >>> selector = VISSA(n_features_to_select=2, n_jobs=2, n_submodels=200)
        >>> selector.fit(X, y)
        >>> selector.get_support()
        array([True, False, False, False, False, True, False, False, False, False])
    """

    def __init__(self,
                 n_features_to_select: int = None,
                 n_submodels: int = 1000,
                 n_jobs: int = 1,
                 n_cv_folds: int = 5,
                 pls: PLSRegression = None,
                 random_state: Union[int, np.random.RandomState] = None):

        super().__init__(n_features_to_select)

        self.pls = pls
        self.n_submodels = n_submodels
        self.n_jobs = n_jobs
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state

    def _produce_submodels(self, var_weights: np.array, random_state):
        n_feats = var_weights.shape[0]
        appearances = np.reshape(np.round(var_weights * self.n_submodels), (-1, 1))

        # populate Binary Sampling Matrix according to weights
        bsm = np.tile(np.arange(1, self.n_submodels + 1).reshape((1, -1)), [n_feats, 1])
        bsm = (bsm <= appearances)

        # create permutation for each a row of the Binary Sampling Matrix
        p = np.arange(self.n_submodels * n_feats)
        random_state.shuffle(p)
        p = np.reshape(p, (n_feats, self.n_submodels))
        p = np.reshape(np.argsort(p, axis=1), (-1,))
        row_selector = np.repeat(np.arange(n_feats), self.n_submodels)

        # permute the Binary Sampling Matrix
        return np.reshape(bsm[(row_selector, p)], (n_feats, self.n_submodels))

    def _evaluate(self, X, y, pls, submodel_index):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(pls,
                                        X,
                                        y,
                                        cv=self.n_cv_folds,
                                        scoring='neg_mean_squared_error')
        return np.mean(cv_scores), submodel_index

    def _evaluate_submodels(self, X, y, pls, bsm):
        submodels = Parallel(n_jobs=self.n_jobs)(delayed(self._evaluate)(X[:, bsm[:, i]],
                                                                         y,
                                                                         PLSRegression() if pls is None else clone(pls),
                                                                         i) for i in range(self.n_submodels))
        return submodels

    def _fit(self, X, y, n_features_to_select):
        random_state = check_random_state(self.random_state)

        # number of top models to used to update the weights of features
        selection_quantile = int(0.05 * self.n_submodels)
        n_subs = self.n_submodels

        score = -10000000
        var_weights = 0.5 * np.ones((X.shape[1],))
        while True:
            cache_score = -10000000
            cache_var_weights = None
            while True:
                bsm = self._produce_submodels(var_weights, random_state)
                # a partial recycling of the bsm is here possible in the following call
                submodels = self._evaluate_submodels(X, y, self.pls, bsm)
                submodels_sorted = sorted(submodels, key=lambda x: -x[0])

                # get submodel indices and scores of best submodels
                top_scores, top_models = list(zip(*(submodels_sorted[: selection_quantile])))
                # get average score of best submodels
                avg_top_scores = np.mean(top_scores)

                if avg_top_scores > cache_score:
                    cache_score = avg_top_scores
                    cache_var_weights = np.sum(bsm[:, top_models], axis=1) / selection_quantile
                else:
                    break

            if cache_score > score:
                score = cache_score
                var_weights = cache_var_weights
            else:
                break

            # early stopping
            if np.sum(var_weights >= ((n_subs - 0.5)/n_subs)) >= n_features_to_select:
                break

        self.weights_ = var_weights
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[np.argsort(-var_weights)[: n_features_to_select]] = True

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_


