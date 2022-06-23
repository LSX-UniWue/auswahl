
import numpy as np

from typing import Union, Dict, List
from sklearn import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, check_scalar

from auswahl._base import PointSelector
from auswahl._vip import VIP
from auswahl._spa import SPA

class VIP_SPA(PointSelector):

    """
        TODO
    """

    def __init__(self,
                 n_features_to_select: Union[int, float] = None,
                 n_cv_folds: int = 5,
                 n_jobs: int = 1,
                 pls: PLSRegression = None,
                 model_hyperparams: Union[Dict, List[Dict]] = None):
        super().__init__(n_features_to_select, model_hyperparams, n_cv_folds)

        self.vip = VIP(n_features_to_select=n_features_to_select,
                       pls=pls,
                       n_cv_folds=n_cv_folds, model_hyperparams=model_hyperparams)
        self.spa = SPA(n_features_to_select=n_features_to_select,
                       n_cv_folds=n_cv_folds, n_jobs=n_jobs,
                       pls=pls, model_hyperparams=model_hyperparams)

    def _fit(self, X, y, n_features_to_select):
        self.vip.fit(X, y)
        # Mask all features with a VIP score below a threshold: implement a shoulder detection here
        mask = self.vip.vips_ > 0.3
        # Call SPA on masked data
        self.spa.fit(X, y, mask=mask)

        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[self.spa.get_support(indices=True)] = True
        self.best_model_ = self.spa.best_model_

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_