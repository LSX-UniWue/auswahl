import warnings
from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._base import PointSelector


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
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 random_state: Union[int, np.random.RandomState] = None):

        super().__init__(n_features_to_select, model_hyperparams=model_hyperparams, n_cv_folds=n_cv_folds,
                         random_state=random_state, n_jobs=n_jobs)

        self.pls = pls
        self.n_submodels = n_submodels

    def _evaluate_submodels(self, X, y, bsm):
        submodels = Parallel(n_jobs=self.n_jobs)(delayed(self._evaluate)(X[:, bsm[:, i]],
                                                                         y,
                                                                         self.pls,
                                                                         True,
                                                                         i) for i in range(self.n_submodels))
        return submodels

    def _produce_submodels_adapted(self, var_weights: np.array, n_features_to_select, n_submodels, random_state):

        """
            Produces an adapted binary sampling matrix generating submodels containing always at least
            self.n_features_to_select features. For large numbers of submodels (>= 1000) the deviation
            from the statistical properties of the sampling matrices as described by Deng et al. is negligible.
        """
        random_mask = random_state.rand(n_submodels, var_weights.size)
        var_weights = var_weights.reshape(1, -1)
        # certainly selected features have a score <= 0
        selection_scores = random_mask - var_weights

        sorted_score_indices = np.argsort(selection_scores,
                                          axis=-1) + 1  # shift the feature indices to [1...n_features]

        # retrieve for each submodel the number of selected features
        cut_off = np.sum(selection_scores <= 0, axis=-1)
        # ensure, that at least n_features_to_select features are selected
        selected_cut_off = np.max(np.stack([cut_off, n_features_to_select * np.ones_like(cut_off)],
                                           axis=-1),
                                  axis=-1)
        # mask non-selected features
        retrieval_mask = np.tile(np.arange(var_weights.size).reshape(1, -1), reps=(n_submodels, 1))
        retrieval_mask = retrieval_mask < np.expand_dims(selected_cut_off, axis=-1)
        selection = sorted_score_indices * retrieval_mask

        # construct binary sampling matrix
        selection = np.nonzero(selection)  # get the positions of the non-masked features
        features = sorted_score_indices[selection]  # translate the non-masked positions to actual feature indices

        bsm = np.zeros((var_weights.size, n_submodels), dtype='int')
        # complement the feature indices with their respective submodel index
        selection = (features - 1, selection[0])  # Undo the shift of the feature indices
        bsm[selection] = 1
        return bsm

    def _produce_submodels_org(self, var_weights: np.array, n_submodels, random_state):

        """
            Returns a binary sampling matrix as described by Deng et al.
        """
        n_feats = var_weights.shape[0]
        appearances = np.reshape(np.round(var_weights * n_submodels), (-1, 1))

        # populate Binary Sampling Matrix according to weights
        bsm = np.tile(np.arange(1, n_submodels + 1).reshape((1, -1)), [n_feats, 1])
        bsm = (bsm <= appearances)

        # create permutation for each a row of the Binary Sampling Matrix
        p = np.arange(n_submodels * n_feats)
        random_state.shuffle(p)
        p = np.reshape(p, (n_feats, n_submodels))
        p = np.reshape(np.argsort(p, axis=1), (-1,))
        row_selector = np.repeat(np.arange(n_feats), n_submodels)

        # permute the Binary Sampling Matrix
        bsm = np.reshape(bsm[(row_selector, p)], (n_feats, n_submodels))

        # count variables per submodel and drop submodels with less than the required number of features
        valid_submodels = np.sum(bsm, axis=0) >= self.n_features_to_select
        return bsm[:, valid_submodels]

    def _yield_best_weights(self, X, y, n_features_to_select, n_submodels, random_state, selection_quantile):
        best_score = -10000000
        best_model = None
        best_variables = None
        # initialize with an equal distribution across all features
        var_weights = 0.5*np.ones(X.shape[1])
        for _ in range(100):
            # produce weighted binary sampling matrix
            bsm = self._produce_submodels_adapted(var_weights, n_features_to_select, n_submodels, random_state)

            # score submodels
            submodels = self._evaluate_submodels(X, y, bsm)
            submodels_sorted = sorted(submodels, key=lambda x: -x[0])

            # get submodel indices and scores of best submodels
            top_scores, top_models, top_submodels = list(zip(*(submodels_sorted[: selection_quantile])))
            # get average score of best submodels
            avg_top_scores = np.mean(top_scores)

            if avg_top_scores > best_score:
                best_score = avg_top_scores
                best_model = top_models[0]
                best_variables = np.nonzero(bsm[:, top_submodels[0]])[0]
                var_weights = np.sum(bsm[:, top_submodels], axis=1) / selection_quantile
            else:
                break

        return best_score, best_model, best_variables, var_weights[best_variables]

    def _fit(self, X, y, n_features_to_select):
        random_state = check_random_state(self.random_state)

        # number of top models to used to update the weights of features
        selection_quantile = int(0.05 * self.n_submodels)
        n_subs = self.n_submodels

        top_score = -10000000
        selected_variables = np.arange(X.shape[1])
        selected_variables_weight = np.zeros(X.shape[1])
        for _ in range(100):
            score, best_model, selection, weights = self._yield_best_weights(X[:, selected_variables],
                                                                 y,
                                                                 n_features_to_select,
                                                                 n_subs,
                                                                 random_state,
                                                                 selection_quantile)

            if score > top_score:
                top_score = score
                selected_variables = selected_variables[selection]
                selected_variables_weight = weights
                self.best_model_ = best_model
            else:
                break

            # required number of features selected
            if selected_variables.size == n_features_to_select:
                break

        self.weights_ = np.zeros(X.shape[1])
        self.weights_[selected_variables] = selected_variables_weight

        self.support_ = np.zeros((X.shape[1],)).astype('bool')
        self.support_[selected_variables[np.argsort(-selected_variables_weight)][:n_features_to_select]] = True

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_


