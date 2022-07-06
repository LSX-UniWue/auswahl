import numpy as np
import pytest

from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression

from auswahl import VIP, IPLS
from auswahl.benchmarking import deng_score, benchmark


def _benchmark_interface_reduced_dof(data=[(np.zeros((50, 100)), np.zeros((50,)), 'test_dataset')],
                                     n_runs=1,
                                     train_size=0.9,
                                     reg_metrics=[mean_squared_error],
                                     stab_metrics=[deng_score],
                                     n_features=[1],
                                     n_intervals=[1],
                                     interval_widths=[1],
                                     methods=[VIP(n_features_to_select=10),
                                              IPLS(n_intervals_to_select=1, interval_width=10, n_jobs=1)]):

    _ = benchmark(data=data,
                  n_runs=n_runs,
                  train_size=train_size,
                  test_model=PLSRegression(n_components=1),
                  reg_metrics=reg_metrics,
                  stab_metrics=stab_metrics,
                  methods=methods,
                  verbose=False,
                  random_state=123456,
                  n_features=n_features,
                  n_intervals=n_intervals,
                  interval_widths=interval_widths)


def test_n_features_exceptions():

    vip = VIP(n_features_to_select=10)
    ipls = IPLS(n_intervals_to_select=1, interval_width=10, n_jobs=1)

    params = [
        # no interval configuration provided for IntervalSelector
        {'n_features': [1], 'n_intervals': None, 'interval_widths': None, 'methods': [ipls]},
        # incomplete specification of the total number of features
        {'n_features': None, 'n_intervals': [1], 'interval_widths':None, 'methods': [ipls, vip]},
        # incomplete specification of the total number of features
        {'n_features': None, 'n_intervals': None, 'interval_widths': [1], 'methods': [vip]},
        # inconsistent total number of features
        {'n_features':[10, 12], 'n_intervals': [1, 1], 'interval_widths':[10, 13], 'methods': [vip, ipls]},
        # inconsistent lengths
        {'n_features': [10, 11, 12], 'n_intervals': [1, 1], 'interval_widths':[10, 11], 'methods': [vip, ipls]},
        # inconsistent lengths
        {'n_features': None, 'n_intervals':[1, 1], 'interval_widths':[10, 11, 12], 'methods': [vip]},
        # n_intervals inconsistent with n_features (divisibility)
        {'n_features': [10, 12], 'n_intervals': [1, 5], 'interval_widths':None, 'methods': [vip, ipls]}
    ]

    for i, param in enumerate(params):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(**param)


def test_dataset_exceptions():

    x = np.zeros((50, 100))
    y = np.zeros((50,))

    params = [{'data': [(x, y, 'test'), (x, y, 'test')]},  # non-unique names
              {'data': [(x, y)]},  # not all fields specified
              {'data': [(10, 11, 'test')]},  # wrong data type
              {'data': [(x, y, 10)]}]  # wrong data type

    for i, param in enumerate(params):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(**param)


def test_metrics():

    params = [{'reg_metrics': [mean_squared_error, mean_squared_error]},  # names not unique
              {'stab_metrics': [deng_score, deng_score]},  # names not unique
              {'reg_metrics': []}]  # no regression metric specified

    for i, param in enumerate(params):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(**param)


def test_train_size():

    data = (np.zeros((10, 100)), np.zeros((10,)), 'test_dataset')

    params = [{'train_size': [0.5, 0.5]},  # length not matching #datasets
              {'train_size': 0},  # out of valid range
              {'train_size': 1},   # out of valid range
              {'data': data, 'train_size': 0.95}]  # invalid train_size w.r.t. dataset size

    for i, param in enumerate(params):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(**param)