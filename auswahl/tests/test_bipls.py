import numpy as np
import pytest

from auswahl import BiPLS


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.randn(100, 50)
    y = X[:, 20:25] * np.array([[-1, 0, 5, 0, -4]]) + X[:, 45:50] * np.array([[3, 0, 1, 0, -4]])
    y = np.sum(y, axis=1)
    return X, y


def test_bipls(data):
    X, y = data
    n_intervals_to_select = 2
    interval_width = 5
    bipls = BiPLS(n_intervals_to_select=n_intervals_to_select, interval_width=interval_width)

    bipls.fit(X, y)
    assert len(bipls.support_) == X.shape[1]
    assert sum(bipls.support_) == n_intervals_to_select * interval_width
    assert sum(bipls.support_[np.r_[20:25, 45:50]]) == 10

    assert bipls.rank_.max() == 1
    assert bipls.rank_.min() == 0
    assert (bipls.rank_ == 1).sum() <= (n_intervals_to_select * interval_width)

    X_t = bipls.transform(X)
    assert X_t.shape[1] == n_intervals_to_select * interval_width


def test_dimension_mismatch(data):
    X, y = data
    X = np.concatenate([X, np.random.randn(100, 3)], axis=1)
    n_intervals_to_select = 2
    interval_width = 5

    bipls = BiPLS(n_intervals_to_select=n_intervals_to_select, interval_width=interval_width)

    bipls.fit(X, y)
    assert len(bipls.support_) == X.shape[1]
    assert sum(bipls.support_) == n_intervals_to_select * interval_width
    assert sum(bipls.support_[np.r_[20:25, 45:50]]) == 10

    X_t = bipls.transform(X)
    assert X_t.shape[1] == n_intervals_to_select * interval_width
