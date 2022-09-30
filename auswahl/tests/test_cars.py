import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.cross_decomposition import PLSRegression

from auswahl import CARS


@pytest.fixture
def data():
    np.random.seed(1337)
    # high sample count to avoid random correlation for testing purposes
    X = np.random.randn(200, 10)
    y = 10 * X[:, 1] - 10 * X[:, 5]
    return X, y


def test_cars(data):
    X, y = data
    selector = CARS(n_features_to_select=2, n_jobs=2, n_cars_runs=20, pls=PLSRegression(n_components=1))

    selector.fit(X, y)
    assert len(selector.support_) == X.shape[1]
    assert sum(selector.support_) == 2
    assert_array_equal(selector.support_, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    X_t = selector.transform(X)
    assert X_t.shape[1] == 2
    assert_array_almost_equal(X[:, [1, 5]], X_t)


def test_cars_hyperparams(data):
    X, y = data
    selector = CARS(n_features_to_select=2, n_jobs=2, n_cars_runs=20, pls=PLSRegression(n_components=1),
                    model_hyperparams={'n_components': [1, 2]})

    selector.fit(X, y)
    assert len(selector.support_) == X.shape[1]
    assert sum(selector.support_) == 2
    assert_array_equal(selector.support_, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    X_t = selector.transform(X)
    assert X_t.shape[1] == 2
    assert_array_almost_equal(X[:, [1, 5]], X_t)
