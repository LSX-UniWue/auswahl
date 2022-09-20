from typing import Union, Dict, List

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted
from numpy.random import RandomState

from ._base import PointSelector, FeatureDescriptor
from ._spa import SPA
from ._vip import VIP


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
        super().__init__(n_features_to_select, model_hyperparams, n_cv_folds, n_jobs=n_jobs)

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

    def reparameterize(self, feature_descriptor: FeatureDescriptor):
        self.n_features_to_select = feature_descriptor.get_configuration_for(self)
        self.vip.reparameterize(feature_descriptor)
        self.spa.reparameterize(feature_descriptor)

    def reseed(self, seed: Union[int, RandomState]):
        self.vip.reseed(seed)
        self.spa.reseed(seed)

    def rethread(self, n_jobs: int):
        self.vip.rethread(n_jobs)
        self.spa.rethread(n_jobs)
