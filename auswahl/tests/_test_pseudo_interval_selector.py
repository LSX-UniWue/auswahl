import numpy as np
import pytest

from auswahl import VIP, PseudoIntervalSelector

@pytest.fixture
def data():
    X = np.random.randn(100, 50)
    y = X[:, 20:25] * np.array([[-1, 0, 5, 0, -4]]) + X[:, 45:50] * np.array([[3, 0, 1, 0, -4]])
    y = np.sum(y, axis=1)
    return X, y


def test_pseudo_interval_selector(data):
    X, y = data
    n_intervals_to_select = 2
    interval_width = 5
    inter_vip = PseudoIntervalSelector(selector=VIP(n_features_to_select=1),
                                       n_intervals_to_select=n_intervals_to_select, interval_width=interval_width)

    inter_vip.fit(X, y)
    assert len(inter_vip.support_) == X.shape[1]
    assert sum(inter_vip.support_) == n_intervals_to_select * interval_width
    assert sum(inter_vip.support_[[20, 22, 24, 45, 47, 49]]) == 6

    X_t = inter_vip.transform(X)
    assert X_t.shape[1] == n_intervals_to_select * interval_width
