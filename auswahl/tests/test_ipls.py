import numpy as np
import pytest

from auswahl import IPLS


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.randn(100, 50)
    y = 5 * X[:, 20] - 2 * X[:, 21] + 8 * X[:, 23]
    return X, y


def test_ipls(data):
    X, y = data
    interval_width = 5
    ipls = IPLS(interval_width=interval_width)

    ipls.fit(X, y)
    assert len(ipls.support_) == X.shape[1]
    assert sum(ipls.support_) == interval_width
    assert sum(ipls.support_[[20, 21, 23]]) == 3

    X_t = ipls.transform(X)
    assert X_t.shape[1] == interval_width
