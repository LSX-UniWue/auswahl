import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from auswahl import SPA


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.rand(15, 100)
    y = 15 * X[:, 10] - 2 * X[:, 20]
    return X, y


@pytest.fixture
def ortho_data():
    np.random.seed(1337)
    X = np.array([[1, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1]])
    X = X / np.linalg.norm(X, ord=2, axis=0)
    y = np.random.rand(4)
    return X, y


def test_spa(data):
    X, y = data
    spa = SPA(n_features_to_select=2, n_cv_folds=2)
    spa.fit(X, y)
    assert len(spa.support_) == X.shape[1]
    assert sum(spa.support_) == 2
    assert spa.support_[10]


def test_orthogonality(ortho_data):
    X, y = ortho_data
    spa = SPA(n_features_to_select=3, n_cv_folds=2)

    spa.fit(X, y)
    assert len(spa.support_) == X.shape[1]
    assert sum(spa.support_) == 3
    selected = np.compress(spa.support_, X, axis=1)
    assert_array_almost_equal(np.transpose(selected) @ selected, np.eye(sum(spa.support_), dtype='float'))
