import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from auswahl import VIP, IPLS
from auswahl.benchmarking import DengScore, benchmark


@pytest.fixture
def data():
    X = np.zeros((50, 10))
    y = np.zeros((50,))
    name = 'test_dataset'
    test_size = 0.2
    return X, y, name, test_size


def _benchmark_interface_reduced_dof(data,
                                     features=None,
                                     n_runs=1,
                                     reg_metrics=None,
                                     stab_metrics=None,
                                     methods=None):
    reg_metrics = [mean_squared_error] if reg_metrics is None else reg_metrics
    stab_metrics = [DengScore] if stab_metrics is None else stab_metrics
    features = [1] if features is None else features
    methods = [VIP(n_features_to_select=10)] if methods is None else methods

    _ = benchmark(data=data,
                  features=features,
                  n_runs=n_runs,
                  reg_metrics=reg_metrics,
                  stab_metrics=stab_metrics,
                  methods=methods,
                  verbose=False,
                  random_state=123456)


def test_n_features_exceptions(data):
    vip = VIP(n_features_to_select=10)
    ipls = IPLS(n_intervals_to_select=1, interval_width=10, n_jobs=1)

    params = [
        {'features': [], 'methods': [vip]},  # nothing given for PointSelector
        {'features': [], 'methods': [ipls]},  # nothing given for IntervalSelector

        {'features': [1], 'methods': [ipls]},  # Point given for IntervalSelector

        {'features': [(1,)], 'methods': [vip]},  # wrong tuple given for PointSelector
        {'features': [(1,)], 'methods': [ipls]},  # wrong tuple given for IntervalSelector
        {'features': [(1, 2, 3)], 'methods': [ipls]},  # wrong tuple given for IntervalSelector

        {'features': [10, (2, 5)], 'methods': [vip, ipls]}  # mixed PointSelector and IntervalSelector
    ]

    for i, param in enumerate(params):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(data=data, **param)


def test_dataset_exceptions(data):
    X, y, name, test_size = data

    erroneous_data = [
        [data, data],  # non-unique names
        [(X, y, name)],  # not all fields specified
        [(10, 11, name, 0.2)],  # wrong data type
        [(X, y, 10, 0.2)],  # wrong data type
        [(X, y, name, '0.2')]  # wrong data type
    ]

    for i, d in enumerate(erroneous_data):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(data=d)


def test_metrics(data):
    params = [
        {'reg_metrics': [mean_squared_error, mean_squared_error]},  # names not unique
        {'stab_metrics': [DengScore, DengScore]},  # names not unique
        {'reg_metrics': []},  # no regression metric specified
    ]

    for i, param in enumerate(params):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(data=data, **param)


def test_train_size(data):
    X, y, name, test_size = data

    erroneous_data = [
        [(X, y, name, 0.0)],  # test_size equals zero
        [(X, y, name, 1.0)],  # test_size equals n_samples
        [(X, y, name, 0.01)],  # test_size higher than 0.0 but to small
        [(X, y, name, 0.99)],  # test_size lower than 1.0 but to large
        [(X, y, name, 1.1)]  # test_size to large
    ]

    for i, d in enumerate(erroneous_data):
        print(f'{i}')
        with pytest.raises(ValueError):
            _benchmark_interface_reduced_dof(data=d)
