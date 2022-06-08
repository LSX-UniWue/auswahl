import numpy as np
import time

from numpy.random import RandomState
from typing import Union, List, Tuple, Callable
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn import clone

from auswahl import PointSelector, IntervalSelector
from benchmark.util._data_handling import BenchmarkPOD
from benchmark.util._metrics import mean_std_statistics


class Speaker:

    """
        Pretty printing facility
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def announce(self, level: int, message: str):
        if self.verbose:
            print("    " * level + message)


def _check_feature_interval_consistency(methods, n_features, n_intervals, interval_widths):

    """
            TODO: streamline the consistency checks (feels non-optimal)
            Check consistency of parameters passed to the benchmarking function.
            Raises exceptions

        Parameters
        ----------
        methods: List[Union[PointSelector, IntervalSelector]]
            list of PointSelectors and IntervalSelectors to be benchmarked
        n_features: List[int]
            list of n_features to be benchmarked
        n_intervals: List[int]
            list of n_intervals to be benchmarked
        interval_widths: List[int]
            list of interval_widths to be benchmarked

        Returns
        -------
        n_features: List[int]
            n_features if n_features is not None, else n_features is inferred from n_intervals and interval_width

    """

    if n_intervals is None:
        for method in methods:
            if isinstance(method, IntervalSelector):
                raise ValueError("Number of intervals for IntervalSelectors is not specified")
        if n_features is None:
            raise ValueError("If n_intervals is not specified, n_features must be specified")
    else:

        if interval_widths is None:

            if n_features is None:
                raise ValueError("n_intervals has been specified. "
                                 "The specification of n_features or interval_width is additionally required")
            else:
                if len(n_features) != len(n_intervals):
                    raise ValueError("The length of the lists n_features and n_intervals are required to be equal."
                                     f'Got {len(n_features)} and {n_intervals}')
                else:
                    if not np.all(np.logical_not(np.mod(np.array(n_features), np.array(n_intervals)))):
                        raise ValueError("n_features are requires to be divisible by n_intervals")
        else:
            if len(interval_widths) != len(n_intervals):
                raise ValueError("The length of the lists interval_width and n_intervals are required to be equal."
                                 f'Got {len(interval_widths)} and {n_intervals}')
            if n_features is not None:
                if not np.all(np.array(n_features) == np.array(interval_widths) * np.array(n_intervals)):
                    raise ValueError("The total number of features is inconsistent between the specification of n_features"
                                     "and n_intervals in conjunction with interval_width")
            else:
                return [interval_widths[i] * n_intervals[i] for i in range(len(n_intervals))]

    return n_features


def _parameterize(method, n_features, n_intervals, interval_widths, seed, index):
    """
            Reconfigure parameterization of methods.

        Parameters
        ----------
        method: Union[PointSelector, IntervalSelector]
            method to be reparameterized
        n_features: List[int]
            list of n_features to be benchmarked
        n_intervals: List[int]
            list of n_intervals to be benchmarked
        interval_widths: List[int]
            list of interval_widths to be benchmarked
        seed: int
            seed to be used by the method
        index: int
            position indicator of the parameters to be used for reparameterization in the lists passed to the function

    """
    if isinstance(method, PointSelector):
        method.n_features_to_select = n_features[index]
    else:
        method.n_intervals_to_select = n_intervals[index]
        if interval_widths is None:
            #infer interval_width
            method.interval_width = n_features[index] // n_intervals[index]
        else:
            method.interval_width = interval_widths[index]

    # reparameterize the method seed, if the method is probabilistic
    if hasattr(method, 'random_state'):
        method.random_state = seed


def _sanitize_n_features(n_features, n_intervals, interval_widths):

    """
        Update: this is not valid
        TODO: update data handling keys
        -> the same number of features can be selected through different number of intervals and respective interval_widths
    """
    _, indices = np.unique(n_features, return_index=True)
    n_features = n_features[indices]
    n_intervals = n_intervals[indices]
    interval_widths = interval_widths[indices]

    return n_features, n_intervals, interval_widths


def _check_name_uniqueness(name_list: List[str], identifier):

    """
        Check the uniqueness of names

        Parameters
        ----------
        name_list: List[str]
            list of names, whose uniqueness is to be checked
        identifier: str
            name of the list for reference in a possible exception to be raised
    """
    if len(name_list) != np.unique(name_list).size:
        raise ValueError('The names in {identifier} need to be unique')


def _unpack_methods(methods):
    """

        Decomposes the method argument passed to function benchmark into a list of selector methods and identifiers.
        Checks types underway.

    Parameters
    ----------
    methods: List[Union[PointSelector, IntervalSelector, Tuple[Union[PointSelector, IntervalSelector], str]]]
        list of feature selection methods or tuples of feature selection methods and their identifiers

    Returns
    -------
    tuple: Tuple[List[Union[PointSelector, IntervalSelector]], List[str]]

    """
    selectors = []
    names = []
    if type(methods) != list:
        methods = [methods]

    for method in methods:
        if isinstance(method, tuple):
            if isinstance(method[0], (PointSelector, IntervalSelector)):
                selectors.append(method[0])
            else:
                raise ValueError(f'Expected first element in {method} to be of type Union[IntervalSelector, PointSelector]')
            if type(method[1]) == str:
                names.append(method[1])
            else:
                raise ValueError(f'Expected second element in {method} to be of type str')
        else:
            if isinstance(method, (PointSelector, IntervalSelector)):
                selectors.append(method)
                # use class name as default name
                names.append(type(method).__name__)
            else:
                raise ValueError(f'Expected {method} to be of type Union[IntervalSelector, PointSelector]')

    return selectors, names


def _unpack_metrics(metrics):
    """

            Decomposes the metrics arguments passed to function benchmark into a list of functions and names.
            Checks types underway.

        Parameters
        ----------
        metrics: List[callable]
            list of metrics

        Returns
        -------
        tuple: Tuple[List[callable], List[str]]

        """
    if not isinstance(metrics, list):
        metrics = [metrics]

    metric_functions = []
    names = []
    for metric in metrics:
        if isinstance(metric, Callable):
            metric_functions.append(metric)
            names.append(metric.__name__)
        else:
            raise ValueError(f'Expected the metric {metric} to be callable')

    return metric_functions, names


def _check_datasets(data):
    """

         Retrieve names of datasets in data.
         Checks types underway.

    Parameters
    ----------
    data: List[Tuple[np.array, np.array, str]]
          list of datasets

    Returns
    -------
        dataset names: List[str]

    """
    if not isinstance(data, list):
        data = [data]

    names = []
    for dataset in data:
        if not isinstance(dataset, (list, tuple)):
            raise ValueError(f'Expected {dataset} to be of type list or tuple')
        if len(dataset) != 3:
            raise ValueError(f'The dataset specification requires three fields: (x, y, name)')
        if not (isinstance(dataset[0], np.ndarray) and isinstance(dataset[1], np.ndarray)):
            raise ValueError(f'The first two elements in the dataset specification (x, y, name) need to be of type np.array ')
        if not isinstance(dataset[2], str):
            raise ValueError(f'The name of a dataset is required to be of type str. Got {type(dataset[2])}')
        names.append(dataset[2])

    return names


def benchmark(data: List[Tuple[np.array, np.array, str]],
              n_runs: int,
              train_size: Union[int, float],
              test_model: BaseEstimator,
              reg_metrics: List[Callable],
              stab_metrics: List[Callable],
              methods: List[Union[PointSelector, IntervalSelector, Tuple[Union[PointSelector, IntervalSelector], str]]],
              random_state: Union[int, RandomState],
              n_features: List[int] = None,
              n_intervals: List[int] = None,
              interval_widths: List[int] = None,
              verbose: bool = True):

    """
        Function performing benchmarking of Interval- and PointSelector feature selectors across
        different datasets and different parameterizations of the selectors.

        Parameters
        ----------
        data
        n_runs
        train_size
        test_model
        reg_metrics
        stab_metrics
        methods
        random_state
        n_features
        n_intervals
        interval_widths
        verbose

        Returns
        -------
        pod: BenchmarkPOD
            TODO

    """

    n_features = _check_feature_interval_consistency(methods, n_features, n_intervals, interval_widths)

    speaker = Speaker(verbose)

    dataset_names = _check_datasets(data)
    reg_metrics, reg_metric_names = _unpack_metrics(reg_metrics)
    stab_metric, stab_metric_names = _unpack_metrics(stab_metrics)
    methods, method_names = _unpack_methods(methods)

    _check_name_uniqueness(dataset_names, "datasets")
    _check_name_uniqueness(method_names,  "methods")
    _check_name_uniqueness(reg_metric_names, "reg_metrics")
    _check_name_uniqueness(stab_metric_names, "stab_metrics")


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
                    _parameterize(method, n_features, n_intervals, interval_widths, seed, i)

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
                    print(f'Measred {end-start}')

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
