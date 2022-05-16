import numpy as np
import time

from numpy.random import RandomState
from typing import Union, List, Tuple
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn import clone

from auswahl import PointSelector, IntervalSelector
from benchmark.util._data_handling import BenchmarkPOD
from benchmark.util._metrics import mean_std_statistics


class Speaker:

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def announce(self, level: int, message: str):
        if self.verbose:
            print("    " * level + message)

def _check_feature_interval_consistency(methods, n_features, n_intervals):
    if n_intervals is None:
        for method in methods:
            if isinstance(method, IntervalSelector):
                raise ValueError("Number of intervals for IntervalSelectors not specified")
    else:
        if len(n_features) != len(n_intervals):
            raise ValueError("The length of the lists n_features and n_intervals are required to be equal."
                             f'Got {len(n_features)} and {n_intervals}')
        else:
            if not np.all(np.logical_not(np.mod(np.array(n_features), np.array(n_intervals)))):
                raise ValueError("n_features are requires to be divisible by n_intervals")

def _parameterize(method, n_features, n_intervals, seed, index):
    if isinstance(method, PointSelector):
        method.n_features_to_select = n_features[index]
    else:
        method.n_intervals_to_select = n_intervals[index]
        method.interval_width = n_features[index] // n_intervals[index]
    if hasattr(method, 'random_state'):
        method.random_state = seed

def benchmark(data: List[Tuple[np.array, np.array, str]],
              n_features: List[int],
              n_runs: int,
              train_size: Union[int, float],
              test_model: BaseEstimator,
              reg_metrics: List,
              stab_metrics: List,
              methods: List[Union[PointSelector, IntervalSelector]],
              random_state: Union[int, RandomState],
              n_intervals: List[int] = None,
              verbose: bool = True):

    _check_feature_interval_consistency(methods, n_features, n_intervals)

    speaker = Speaker(verbose)

    dataset_names = [tup[2] for tup in data]
    method_names = [type(method).__name__ for method in (methods if type(methods) == list else [methods])]
    reg_metric_names = [metric.__name__ for metric in (reg_metrics if type(reg_metrics) == list else [reg_metrics])]
    stab_metric_names = [stab.__name__ for stab in (stab_metrics if type(stab_metrics) == list else [stab_metrics])]

    random_state = check_random_state(random_state)

    pod = BenchmarkPOD(dataset_names,
                       method_names,
                       n_features,
                       reg_metric_names,
                       stab_metric_names,
                       n_runs
                       )

    pod.register_meta(data)

    for x, y, dataset_name in data:
        speaker.announce(level=0, message=f'Started benchmark for dataset {dataset_name}')
        for i, n in enumerate(n_features):
            speaker.announce(level=1, message=f'Started cycle with {n} features to select:')
            for r in range(n_runs):
                speaker.announce(level=2, message=f'Started run: {r}')
                seed = random_state.randint(0, 1000000)
                train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=train_size, random_state=seed)
                for method_name, method in zip(method_names, methods):
                    speaker.announce(level=3, message=f'started method {method_name}')
                    _parameterize(method, n_features, n_intervals, seed, i)

                    start = time.process_time()
                    method.fit(train_x, train_y)
                    end = time.process_time()

                    support = method.get_support(indices=True)

                    test_regressor = clone(test_model)
                    test_regressor.fit(train_x[:, support], train_y)
                    prediction = test_regressor.predict(test_x[:, support])

                    for metric_name, metric in zip(reg_metric_names, reg_metrics):
                        pod.register_regression(metric(test_y, prediction),
                                                dataset_name,
                                                method_name,
                                                n,
                                                metric_name,
                                                'samples',
                                                r)

                    pod.register_measurement(end-start,
                                             dataset_name,
                                             method_name,
                                             n,
                                             'samples',
                                             r)

                    pod.register_selection(dataset_name,
                                           method_name,
                                           n,
                                           r,
                                           support)

    # mean and std over all regression metrics, runs and datasets
    mean_std_statistics(pod)

    # stability scores
    for metric_name, metric in zip(stab_metric_names, stab_metrics):
        metric(pod)

    return pod
