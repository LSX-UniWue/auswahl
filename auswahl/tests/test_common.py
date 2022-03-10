import pytest
from sklearn.utils.estimator_checks import check_estimator

from auswahl import VIP, MCUVE, RandomFrog, SPA, IPLS


@pytest.mark.parametrize("estimator", [IPLS(), SPA(), VIP(), MCUVE(), RandomFrog(n_iterations=10)])
def test_all_estimators(estimator):
    return check_estimator(estimator)
