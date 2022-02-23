import pytest
from sklearn.utils.estimator_checks import check_estimator

from auswahl import VIP, MCUVE, RandomFrog


@pytest.mark.parametrize("estimator", [VIP(), MCUVE(), RandomFrog(n_iterations=10)])
def test_all_estimators(estimator):
    return check_estimator(estimator)
