from typing import Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils.validation import check_is_fitted

from ._base import FeatureDescriptor
from ._base import IntervalSelector, Convertible, SpectralSelector
from .util import optimize_intervals


class PseudoIntervalSelector(IntervalSelector):
    """PseudoIntervalSelector transforms a PointSelector subclassing :class:`~auswahl.Convertible` into an
    IntervalSelector. Given the feature scores calculated by the wrapped :class:`~auswahl.PointSelector`, an optimal
    (max total score) interval placement is calculated using :func:`~auswahl.optimize_intervals`.

    Parameters
    ----------
    selector: Convertible
        Instance of a PointSelector subclassing Convertible.

    n_intervals_to_select : int, default=None
        Number of intervals to select.

    interval_width : int or float, default=None
        Number of features that form an interval.
    """

    def __init__(self,
                 selector: Convertible,
                 n_intervals_to_select: int = None,
                 interval_width: Union[int, float] = None):
        super().__init__(n_intervals_to_select=n_intervals_to_select, interval_width=interval_width)

        if not isinstance(selector, Convertible):
            raise ValueError('PseudoIntervalSelector requires a selector subclassing Convertible.')
        if not isinstance(selector, SpectralSelector):
            raise ValueError('PseudoIntervalSelector requires a selector subclassing SpectralSelector.')

        self.selector = selector

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        self.selector.reparameterize(FeatureDescriptor(key=(n_intervals_to_select, interval_width)))
        self.selector.fit(X, y)
        scores = self.selector.get_feature_scores()
        _, interval_starts = optimize_intervals(n_intervals_to_select, interval_width, feature_scores=scores)
        intervals = np.reshape(interval_starts, (-1, 1)) + np.arange(interval_width).reshape((1, -1))
        self.support_ = np.zeros((X.shape[1],), dtype=bool)
        self.support_[intervals.flatten()] = 1
        self.best_model_ = self.selector.get_best_estimator()

    def reparameterize(self, feature_descriptor: FeatureDescriptor):
        self.n_intervals_to_select, self.interval_width = feature_descriptor.get_configuration_for(self)
        self.selector.reparameterize(feature_descriptor)

    def reseed(self, seed: Union[int, RandomState]):
        self.selector.reseed(seed)

    def rethread(self, n_jobs: int):
        self.selector.rethread(n_jobs)
