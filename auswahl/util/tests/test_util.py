import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.cross_decomposition import PLSRegression

from auswahl.util._pls_utils import get_coef_from_pls


@pytest.fixture
def data():
    np.random.seed(1337)
    X = np.random.randn(100, 10)
    y = 5 * X[:, 0] - 2 * X[:, 5]
    return X, y


def test_coef(data):
    X, y = data
    pls = PLSRegression()
    pls.fit(X, y)

    coef_ = get_coef_from_pls(pls)
    assert_array_almost_equal(pls.coef_.T, coef_)
