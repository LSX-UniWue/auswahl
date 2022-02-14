import pytest

from sklearn.utils.estimator_checks import check_estimator

from auswahl import VIP


@pytest.mark.parametrize("estimator", [VIP()])
def test_all_estimators(estimator):
    return check_estimator(estimator)
