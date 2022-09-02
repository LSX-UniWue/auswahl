from typing import Union

from ._base import PointSelector, IntervalSelector


class DummyPointSelector(PointSelector):
    """Dummy selector realizing abstract base classes to test their functionality overarchingly.
    """

    def __init__(self, n_features_to_select: int = None):
        super().__init__(n_features_to_select)

    def _fit(self, X, y, n_features_to_select):
        ...

    def _get_support_mask(self):
        return None


class DummyIntervalSelector(IntervalSelector):
    """Dummy selector realizing abstract base classes to test their functionality overarchingly.
    """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None):
        super().__init__(n_intervals_to_select, interval_width)

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        ...

    def _get_support_mask(self):
        return None


class ExceptionalSelector(PointSelector):
    """Dummy selector simply throwing an exception while fitting.
    """

    def __init__(self, n_features_to_select: int = None):
        super().__init__(n_features_to_select)

    def _fit(self, X, y, n_features_to_select):
        raise NotImplementedError("This function is not meaningfully implemented")

    def _get_support_mask(self):
        return None
