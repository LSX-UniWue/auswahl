import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.cross_decomposition import PLSRegression

from auswahl import VIP


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.randn(100, 10)
    y = 5 * X[:, 0] - 2 * X[:, 5]
    return X, y


def test_vip(data):
    X, y = data

    vip = VIP(n_features_to_select=2)

    vip.fit(X, y)
    assert len(vip.support_) == X.shape[1]
    assert sum(vip.support_) == 2
    assert_array_equal(vip.support_, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    X_t = vip.transform(X)
    assert X_t.shape[1] == 2
    assert_array_almost_equal(X[:, [0, 5]], X_t)


def test_sequential_vip(data):
    # Test the VIP implementation against a sequential implementation of the calculation
    X, y = data
    pls = PLSRegression()
    pls.fit(X, y)

    x_scores = pls.transform(X)
    x_weights = pls.x_weights_
    y_loadings = pls.y_loadings_

    vips = []
    for j in range(X.shape[1]):
        weighted_variance, explained_variance = 0, 0
        for a in range(y_loadings.shape[1]):
            explained_variance_ = (y_loadings[0, a] ** 2) * np.dot(x_scores[:, a], x_scores[:, a])
            weighted_variance += explained_variance_ * ((x_weights[j, a] / np.linalg.norm(x_weights[:, a])) ** 2)
            explained_variance += explained_variance_
        vips.append(np.sqrt(X.shape[1] * (weighted_variance / explained_variance)))

    model = VIP(n_features_to_select=2)
    model.fit(X, y)

    assert_array_almost_equal(vips, model.vips_)
