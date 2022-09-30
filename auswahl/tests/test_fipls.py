import numpy as np
import pytest

from auswahl import FiPLS


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.randn(100, 50)
    y = X[:, 20:25] * np.array([[-1, 0, 5, 0, -4]]) + X[:, 45:50] * np.array([[3, 0, 1, 0, -4]])
    y = np.sum(y, axis=1)
    return X, y


def test_fipls(data):
    X, y = data
    n_intervals_to_select = 2
    interval_width = 5
    fipls = FiPLS(n_intervals_to_select=n_intervals_to_select, interval_width=interval_width)

    fipls.fit(X, y)
    assert len(fipls.support_) == X.shape[1]
    assert sum(fipls.support_) == n_intervals_to_select * interval_width
    assert sum(fipls.support_[[20, 22, 24, 45, 47, 49]]) == 6

    X_t = fipls.transform(X)
    assert X_t.shape[1] == n_intervals_to_select * interval_width
