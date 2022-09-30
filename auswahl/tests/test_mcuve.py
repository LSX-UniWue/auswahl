import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from auswahl import MCUVE


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.randn(100, 10)
    y = 5 * X[:, 0] - 2 * X[:, 5]
    return X, y


def test_mcuve(data):
    X, y = data
    selector = MCUVE(n_features_to_select=2)

    selector.fit(X, y)
    assert len(selector.support_) == X.shape[1]
    assert sum(selector.support_) == 2
    assert_array_equal(selector.support_, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    X_t = selector.transform(X)
    assert X_t.shape[1] == 2
    assert_array_almost_equal(X[:, [0, 5]], X_t)


def test_reproducible_run(data):
    X, y = data

    # Passing an int for the random state has to result in the same outcome for each call of the fit method
    selector = MCUVE(n_features_to_select=2, random_state=42)
    stability1 = selector.fit(X, y).stability_.copy()
    stability2 = selector.fit(X, y).stability_
    assert_array_almost_equal(stability1, stability2)

    # If a RandomState is used, two selectors with the same RandomState compute the same outcome after each run
    selector1 = MCUVE(n_features_to_select=2, random_state=np.random.RandomState(42))
    selector2 = MCUVE(n_features_to_select=2, random_state=np.random.RandomState(42))

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert_array_almost_equal(selector1.stability_, selector2.stability_)

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert_array_almost_equal(selector1.stability_, selector2.stability_)
