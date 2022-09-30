import numpy as np
import pytest
from numpy.testing import assert_array_equal

from auswahl import VISSA


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.randn(200, 10)
    y = 5 * X[:, 0] - 2 * X[:, 5]
    return X, y


def test_vissa(data):
    X, y = data
    selector = VISSA(n_features_to_select=2, n_submodels=100)

    selector.fit(X, y)
    assert len(selector.support_) == X.shape[1]
    assert sum(selector.support_) == 2
    assert_array_equal(selector.support_, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    X_t = selector.transform(X)
    assert X_t.shape[1] == 2


def test_reproducible_run(data):
    X, y = data

    # Passing an int for the random state has to result in the same outcome for each call of the fit method
    selector = VISSA(n_features_to_select=2, random_state=42, n_submodels=100)
    weights1 = selector.fit(X, y).frequency_.copy()
    weights2 = selector.fit(X, y).frequency_
    assert_array_equal(weights1, weights2)

    # If a RandomState is used, two selectors with the same RandomState compute the same outcome after each run
    selector1 = VISSA(n_features_to_select=2, random_state=np.random.RandomState(42), n_submodels=100)
    selector2 = VISSA(n_features_to_select=2, random_state=np.random.RandomState(42), n_submodels=100)

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert_array_equal(selector1.support_, selector2.support_)
