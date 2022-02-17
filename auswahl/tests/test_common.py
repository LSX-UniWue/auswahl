import pytest
from sklearn.utils.estimator_checks import check_estimator

from auswahl import VIP, MCUVE


@pytest.mark.parametrize("estimator", [VIP(), MCUVE()])
def test_all_estimators(estimator):
    return check_estimator(estimator)
