import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from auswahl import SPA


@pytest.fixture
def data():
    X = np.array([[1, 0, 0, 3, 0],
                  [0, 1, 0, 0, 2],
                  [0, 0, 1, 0, 2]])
    y = 5 * X[:, 3] - 2 * X[:, 2]
    return X, y


def test_spa(data):
    X, y = data

    spa = SPA(n_features_to_select=2, n_cv_folds=2)

    spa.fit(X, y)
    assert len(spa.support_) == X.shape[1]
    assert sum(spa.support_) == 2
    #assert (spa.support_[3] == 1 and spa.support_[0] == 0)
