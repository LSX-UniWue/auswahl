from typing import Union
from auswahl._base import PointSelector, IntervalSelector


class DummyPointSelector(PointSelector):
    """
        Dummy selector realizing abstract base classes to test their functionality  overarchingly
    """

    def __init__(self, n_features_to_select: int = None):
        super().__init__(n_features_to_select)

    def _fit(self, X, y):
        ...

    def _get_support_mask(self):
        return None

class DummyIntervalSelector(IntervalSelector):
    """
            Dummy selector realizing abstract base classes to test their functionality  overarchingly
    """

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None):
        super().__init__(n_intervals_to_select, interval_width)

    def _fit(self, X, y):
        ...

    def _get_support_mask(self):
        return None