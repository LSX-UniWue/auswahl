import numpy as np
import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression

from auswahl import VIP, IPLS
from benchmark import benchmark, deng_score


def _benchmark_interface_reduced_dof_features(n_features, n_intervals, interval_widths, methods):
    x = np.zeros((50, 100))
    y = np.zeros((50,))
    _ = benchmark(data=[(x, y, 'test_dataset')],
                  n_runs=5,
                  train_size=0.9,
                  test_model=PLSRegression(n_components=1),
                  reg_metrics=[mean_squared_error],
                  stab_metrics=[deng_score],
                  methods=methods,
                  verbose=False,
                  random_state=123456,
                  n_features=n_features,
                  n_intervals=n_intervals,
                  interval_widths=interval_widths
                 )


def _benchmark_interface_reduced_dof_dataset(datasets):

    vip = VIP(n_features_to_select=10)
    ipls = IPLS(n_intervals_to_select=1, interval_width=10, n_jobs=1)

    _ = benchmark(data=datasets,
                  n_runs=5,
                  train_size=0.9,
                  test_model=PLSRegression(n_components=1),
                  reg_metrics=[mean_squared_error],
                  stab_metrics=[deng_score],
                  methods=[vip, ipls],
                  verbose=False,
                  random_state=123456,
                  n_features=[10, 11],
                  n_intervals=[1, 1],
                  )


def test_n_features_exceptions():

    vip = VIP(n_features_to_select=10)
    ipls = IPLS(n_intervals_to_select=1, interval_width=10, n_jobs=1)

    params = [
        # n_features, n_intervals, interval_widths, methods
        [[1], None, None, [ipls]],  # no interval configuration provided for IntervalSelector
        [None, [1], None, [ipls, vip]],  # incomplete specification of the total number of features
        [None, None, [1], [vip]],  # incomplete specification of the total number of features
        [[10, 12], [1, 1], [10, 13], [vip, ipls]],  # inconsistent total number of features
        [[10, 11, 12], [1, 1], [10, 11], [vip, ipls]],  #inconsistent lengths
        [None, [1, 1], [10, 11, 12], [vip]],  # inconsistent lengths
        [[10, 12], [1, 5], None, [vip, ipls]],  #n_intervals inconsistent with n_features (divisibility)

    ]

    for param in params:
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof_features(*param)


def test_dataset_exceptions():

    x = np.zeros((50, 100))
    y = np.zeros((50,))

    params = [[(x, y, 'test'), (x, y, 'test')],  # non-unique names
              [(x, y)],  # not all fields specified
              [(10, 11, 'test')],  # wrong data type
              [(x, y, 10)]]  # wrong data type

    for param in params:
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof_dataset(param)