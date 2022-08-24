import numpy as np
import pytest

from auswahl import IntervalRandomFrog


@pytest.fixture
def data():
    rs = np.random.RandomState(1337)
    X = rs.randn(100, 50)
    y = 5 * X[:, 21] - 2 * X[:, 24] + 3 * X[:, 46] + X[:, 47]
    return X, y


def test_irf(data):
    X, y = data
    n_intervals_to_select = 2
    interval_width = 5

    irf = IntervalRandomFrog(n_intervals_to_select=n_intervals_to_select,
                             interval_width=interval_width,
                             n_iterations=100,
                             random_state=np.random.RandomState(42))
    irf.fit(X, y)

    assert len(irf.support_) == X.shape[1]
    assert sum(irf.support_) == n_intervals_to_select * interval_width
    assert sum(irf.support_[[21, 24, 46, 47]]) == 4

    X_t = irf.transform(X)
    assert X_t.shape[1] == n_intervals_to_select * interval_width
