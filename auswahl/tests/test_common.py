import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from auswahl import VISSA, CARS, VIP, MCUVE, RandomFrog, SPA, IPLS, BiPLS, FiPLS, IntervalRandomFrog
from auswahl._dummy import DummyPointSelector, DummyIntervalSelector


@pytest.mark.parametrize("estimator",
                         [BiPLS(), CARS(), FiPLS(), IPLS(), MCUVE(), RandomFrog(n_iterations=10),
                          IntervalRandomFrog(n_iterations=10), SPA(), VIP(), VISSA(n_submodels=20)])
def test_all_estimators(estimator):
    return check_estimator(estimator)


def test_exceptions_interval_selection():
    x = np.zeros((50, 100))
    y = np.zeros((50,))

    dummies = [
        DummyIntervalSelector(n_intervals_to_select=[2]),  # non-scalar
        DummyIntervalSelector(n_intervals_to_select=0),  # min-val violated
        DummyIntervalSelector(n_intervals_to_select=x.shape[1]),  # max-val violated
        DummyIntervalSelector(n_intervals_to_select=1, interval_width=x.shape[1]),  # max interval width violated
        DummyIntervalSelector(n_intervals_to_select=1, interval_width=-1),  # negative interval width
        DummyIntervalSelector(n_intervals_to_select=2, interval_width=x.shape[1] / 2),  # inconsistent combination
        DummyIntervalSelector(n_intervals_to_select=2, interval_width=0.5)  # inconsistent combination
    ]

    with pytest.raises(TypeError):
        dummies[0].fit(x, y)

    for est in dummies[1:]:
        with pytest.raises(ValueError):
            est.fit(x, y)


def test_exceptions_point_selection():
    x = np.zeros((50, 100))
    y = np.zeros((50,))

    dummies = [
        DummyPointSelector(n_features_to_select=[1]),  # non-scalar
        DummyPointSelector(n_features_to_select=0),  # min-val violated
        DummyPointSelector(n_features_to_select=x.shape[1]),  # max-val violoated
    ]

    with pytest.raises(TypeError):
        dummies[0].fit(x, y)

    for est in dummies[1:]:
        with pytest.raises(ValueError):
            est.fit(x, y)
