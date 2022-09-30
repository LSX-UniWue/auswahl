import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from auswahl import RandomFrog, IntervalRandomFrog


@pytest.fixture
def data():
    np.random.seed(1337)
    rs = np.random.RandomState(1337)
    X = rs.randn(100, 10)
    y = 5 * X[:, 0] - 2 * X[:, 5]
    return X, y


@pytest.fixture
def interval_data():
    np.random.seed(1337)
    X = np.random.randn(100, 50)
    y = 5 * X[:, 21] - 2 * X[:, 24] + 3 * X[:, 46] + X[:, 47]
    return X, y


def test_random_frog(data):
    X, y = data
    n_iterations = 1000

    rf = RandomFrog(n_features_to_select=2, n_iterations=n_iterations, random_state=7331)

    rf.fit(X, y)
    assert len(rf.support_) == X.shape[1]
    assert sum(rf.support_) == 2
    assert all(rf.frequencies_ <= n_iterations)
    assert_array_equal(rf.support_, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    X_t = rf.transform(X)
    assert X_t.shape[1] == 2
    assert_array_almost_equal(X[:, [0, 5]], X_t)


def test_reproducible_run(data):
    X, y = data

    # Passing an int for the random state has to result in the same outcome for each call of the fit method
    selector = RandomFrog(n_features_to_select=2, n_iterations=100, random_state=42)
    frequencies1 = selector.fit(X, y).frequencies_.copy()
    frequencies2 = selector.fit(X, y).frequencies_
    assert_array_equal(frequencies1, frequencies2)

    # If a RandomState is used, two selectors with the same RandomState compute the same outcome after each run
    selector1 = RandomFrog(n_features_to_select=2, n_iterations=100,
                           random_state=np.random.RandomState(42))
    selector2 = RandomFrog(n_features_to_select=2, n_iterations=100,
                           random_state=np.random.RandomState(42))

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert_array_equal(selector1.frequencies_, selector2.frequencies_)

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert_array_equal(selector1.frequencies_, selector2.frequencies_)


def test_interval_random_frog(interval_data):
    X, y = interval_data

    n_intervals_to_select = 2
    interval_width = 5
    n_iterations = 1000
    rf = IntervalRandomFrog(n_intervals_to_select=n_intervals_to_select,
                            interval_width=interval_width,
                            n_iterations=n_iterations,
                            random_state=7331)

    rf.fit(X, y)
    assert len(rf.support_) == X.shape[1]
    assert sum(rf.support_) == 10
    assert all(rf.frequencies_ <= n_iterations)
    assert all(rf.support_[[21, 24, 46, 47]])

    X_t = rf.transform(X)
    assert X_t.shape[1] == n_intervals_to_select * interval_width


def test_reproducible_run_interval_random_frog(interval_data):
    X, y = interval_data

    # Passing an int for the random state has to result in the same outcome for each call of the fit method
    selector = IntervalRandomFrog(n_intervals_to_select=2, interval_width=2, n_iterations=100, random_state=42)
    frequencies1 = selector.fit(X, y).frequencies_.copy()
    frequencies2 = selector.fit(X, y).frequencies_
    assert_array_equal(frequencies1, frequencies2)

    # If a RandomState is used, two selectors with the same RandomState compute the same outcome after each run
    selector1 = IntervalRandomFrog(n_intervals_to_select=2, interval_width=2, n_iterations=100,
                                   random_state=np.random.RandomState(42))
    selector2 = IntervalRandomFrog(n_intervals_to_select=2, interval_width=2, n_iterations=100,
                                   random_state=np.random.RandomState(42))

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert_array_equal(selector1.frequencies_, selector2.frequencies_)

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert_array_equal(selector1.frequencies_, selector2.frequencies_)
