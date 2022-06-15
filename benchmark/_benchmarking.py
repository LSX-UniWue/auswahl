import numpy as np
import time
import copy
import traceback
import json

from numpy.random import RandomState
from typing import Union, List, Tuple, Callable

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn import clone

from auswahl import PointSelector, IntervalSelector
from benchmark.util._data_handling import BenchmarkPOD
from benchmark.util._metrics import mean_std_statistics

from joblib import Parallel, delayed


class Speaker:

    """
        Pretty printing facility
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def announce(self, level: int, message: str):
        if self.verbose:
            print("    " * level + message)


class ErrorLogger:

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.log = dict()

        # meta logging information
        self.dataset = None
        self.features = None


    def set_meta(self, dataset, features):
        self.dataset = dataset
        self.features = features

    def log_error(self, run_index, seed: int, method_name: str, during: str, exception: Exception):
        self.log[f'error {len(self.log) + 1}'] = {'dataset': self.dataset,
                                                  'features': self.features,
                                                  'run': run_index,
                                                  'seed': str(seed),
                                                  'method': method_name,
                                                  'during': during,
                                                  'type': type(exception).__name__,
                                                  'message': str(exception),
                                                  'trace': "".join(traceback.TracebackException.from_exception(exception).format())}

    def write_log(self):
        with open(self.log_file, 'w') as file:
            json.dump(self.log, file, indent=4)


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
        if n_features is None:
            raise ValueError("If n_intervals is not specified, n_features must be specified")
        if interval_widths is None:
            for method in methods:
                if isinstance(method, IntervalSelector):
                    raise ValueError("Number of intervals for IntervalSelectors is not specified")
        else:  # check consistency
            if len(n_features) != len(interval_widths):
                raise ValueError("The length of the lists n_features and interval_widths are required to be equal."
                                 f'Got {len(n_features)} and {interval_widths}')
            if not np.all(np.logical_not(np.mod(np.array(n_features), np.array(interval_widths)))):
                raise ValueError("n_features are requires to be divisible by interval_widths")

            # infer n_intervals
            n_intervals = [n_features[i] // interval_widths[i] for i in range(len(n_features))]

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
            # infer interval_widths
            interval_widths = [n_features[i] // n_intervals[i] for i in range(len(n_intervals))]
        else:
            if len(interval_widths) != len(n_intervals):
                raise ValueError("The length of the lists interval_width and n_intervals are required to be equal."
                                 f'Got {len(interval_widths)} and {n_intervals}')
            if n_features is not None:
                if not np.all(np.array(n_features) == np.array(interval_widths) * np.array(n_intervals)):
                    raise ValueError("The total number of features is inconsistent between the specification of n_features"
                                     "and n_intervals in conjunction with interval_width")
            else:
                # infer n_features
                return [interval_widths[i] * n_intervals[i] for i in range(len(n_intervals))]

    return n_features, n_intervals, interval_widths


def _parameterize(methods, n_features, n_intervals, interval_widths, index):
    """
            Reconfigure parameterization of methods.

        Parameters
        ----------
        method: List[Union[PointSelector, IntervalSelector]]
            method to be reparameterized
        n_features: List[int]
            list of n_features to be benchmarked
        n_intervals: List[int]
            list of n_intervals to be benchmarked
        interval_widths: List[int]
            list of interval_widths to be benchmarked
        index: int
            position indicator of the parameters to be used for reparameterization in the lists passed to the function

    """
    for method in methods:
        if isinstance(method, PointSelector):
            method.n_features_to_select = n_features[index]
        else:
            method.n_intervals_to_select = n_intervals[index]
            method.interval_width = interval_widths[index]


def _reseed(method, seed):
    """
            Update seed of method

    Parameters
    ----------
    method: Union[PointSelector, IntervalSelector]
        method to be re-seeded
    seed: int
        random seed

    """
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


def _unpack_metrics(metrics, compulsory=False):
    """

            Decomposes the metrics arguments passed to function benchmark into a list of functions and names.
            Checks types underway.

        Parameters
        ----------
        metrics: List[Callable[[np.ndarray, np.ndarray], float]]
            list of metrics

        Returns
        -------
        tuple: Tuple[List[Callable], List[str]]

        """
    if not isinstance(metrics, list):
        metrics = [metrics]

    if compulsory and len(metrics) == 0:
        raise ValueError(f'At least one regression metric has to be specified.')

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
        tuple: data if data is a list, else [data]
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

    return data, names


def _check_train_size(train_size, data):

    """
        Sanitize train_size passed to function benchmarking. Check the proper range and scales
        w.r.t. the dataset sizes.

        Parameters
        ----------
        train_size: Union[float, List[float]]
            train_size passed to function benchmark
        data: List[Tuple[np.ndarray, np.ndarray, str]]
            datasets passed to function benchmark

        Returns
        -------
        List[float]
            returns train_size. If need be expanded to match the number of datasets

    """

    if isinstance(train_size, list):
        if len(train_size) != len(data):
            raise ValueError(f'If train_size is specified as list, its length has to match the number of datasets')
        for size in train_size:
            if size <= 0 or size >= 1:
                raise ValueError(f'train_size expected to be in ]0,1[. Got {size}')
    else:
        if train_size <= 0 or train_size >= 1:
            raise ValueError(f'train_size expected to be in ]0,1[. Got {train_size}')
        train_size = [train_size] * len(data)

    #Check if train_size leaves a non-empty set of test data for each dataset
    for i in range(len(train_size)):
        x, _, name = data[i]
        if int(x.shape[0] * (1 - train_size[i])) == 0:
            raise ValueError(f'Given the size of the dataset {name}, the specified train_size {train_size[i]} leaves'
                             f'an empty test set.')
    return train_size


def _check_n_runs(n_runs):

    if not isinstance(n_runs, int):
        raise ValueError(f'n_runs is required to be an integer')
    if n_runs <= 0:
        raise ValueError(f'n_runs is required to be positive')


def _drain_threads(methods):

    """
            Configure the methods to be single-thread operating

    Parameters
    ----------
    methods: List[Union[PointSelector, IntervalSelector]]

    """

    for method in methods:
        if hasattr(method, 'n_jobs'):
            method.n_jobs = 1


def _copy_methods(methods):
    return [copy.deepcopy(method) for method in methods]


def _benchmark_parallel(x: np.array, y: np.array, train_size: float, model: BaseEstimator,
                        methods, method_names, reg_metrics, seed: int, run_index: int):

    methods = _copy_methods(methods)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=train_size, random_state=seed)

    results = dict()
    for method_name, method in zip(method_names, methods):
        results[method_name] = dict()
        _reseed(method, seed)
        try:
            start = time.process_time()
            method.fit(train_x, train_y)
            end = time.process_time()
        except Exception as e:
            results[method_name]['exception'] = (e, run_index, seed, "Fitting of Selector")
        else:
            support = method.get_support(indices=True)
            try:
                test_regressor = clone(model)  # TODO: removal might be possible (data is copied for default joblib multiprocessing)
                test_regressor.fit(train_x[:, support], train_y)
                prediction = test_regressor.predict(test_x[:, support])
            except Exception as e:
                results[method_name]['exception'] = (e, run_index, seed, 'Fitting/prediction of test model')
            else:
                try:
                    metrics = []
                    for metric in reg_metrics:
                        metrics.append(metric(test_y, prediction))
                except Exception as e:
                    results[method_name]['exception'] = (e, run_index, seed, 'Metric evaluation')
                else:
                    results[method_name]['metrics'] = metrics
                    results[method_name]['exec'] = end-start
                    results[method_name]['selection'] = support

    return results


def _pot(pod, dataset_name, feature_index, methods_names, reg_metrics_names, results, logger: ErrorLogger):
    #  the actual index of the threads run is irrelevant
    for i, result in enumerate(results):
        for method in methods_names:
            if 'exception' not in result[method].keys():
                pod.register_measurement(result[method]['exec'], dataset_name, method,
                                         feature_index, 'samples', i)
                pod.register_selection(dataset_name, method,
                                       feature_index, i, result[method]['selection'])
                for j, metric in enumerate(reg_metrics_names):
                    pod.register_regression(result[method]['metrics'][j], dataset_name, method,
                                            feature_index, metric, 'samples', i)
            else:
                exception, run_index, seed, during = result[method]['exception']
                logger.log_error(run_index=run_index, seed=seed, method_name=method, during=during, exception=exception)


def benchmark(data: List[Tuple[np.array, np.array, str]],
              n_runs: int,
              train_size: Union[float, List[float]],
              test_model: BaseEstimator,
              methods: List[Union[PointSelector, IntervalSelector, Tuple[Union[PointSelector, IntervalSelector], str]]],
              reg_metrics: List[Callable[[np.ndarray, np.ndarray], float]],
              random_state: Union[int, RandomState],
              stab_metrics: List[Callable[[np.ndarray, np.ndarray], float]] = [],
              n_features: List[int] = None,
              n_intervals: List[int] = None,
              interval_widths: List[int] = None,
              n_jobs: int = 1,
              error_log_file: str = "./error_log.txt",
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
        n_jobs
        error_log_file
        verbose

        Returns
        -------
        pod: BenchmarkPOD
            TODO

    """

    n_features, n_intervals, interval_widths = _check_feature_interval_consistency(methods, n_features,
                                                                                   n_intervals, interval_widths)

    speaker = Speaker(verbose)
    logger = ErrorLogger(log_file=error_log_file)

    data, dataset_names = _check_datasets(data)
    reg_metrics, reg_metric_names = _unpack_metrics(reg_metrics, compulsory=True)
    stab_metric, stab_metric_names = _unpack_metrics(stab_metrics)
    methods, method_names = _unpack_methods(methods)

    _check_name_uniqueness(dataset_names, "datasets")
    _check_name_uniqueness(method_names,  "methods")
    _check_name_uniqueness(reg_metric_names, "reg_metrics")
    _check_name_uniqueness(stab_metric_names, "stab_metrics")

    train_size = _check_train_size(train_size, data)
    _check_n_runs(n_runs)

    # configure the methods to have a single-thread operation (parallelization is exploited at the benchmarking level)
    _drain_threads(methods)

    random_state = check_random_state(random_state)

    pod = BenchmarkPOD(dataset_names,
                       method_names,
                       n_features,
                       reg_metric_names,
                       stab_metric_names,
                       n_runs
                       )

    pod.register_meta(data)

    # pregenerate seeds, such that the seeding of data splits does not depend on the arguments except random_state
    run_seeds = random_state.randint(0, 1000000, size=n_runs)

    for d, (x, y, dataset_name) in enumerate(data):
        speaker.announce(level=0, message=f'Started benchmark for dataset {dataset_name}')
        for i, n in enumerate(n_features):
            speaker.announce(level=1, message=f'Started cycle with {n} features to select:')
            logger.set_meta(dataset=dataset_name, features=n)

            _parameterize(methods, n_features, n_intervals, interval_widths, i)
            results = Parallel(n_jobs=n_jobs)(delayed(_benchmark_parallel)(x, y, train_size[d], test_model,
                                                                           methods, method_names, reg_metrics,
                                                                           run_seeds[r], r)
                                              for r in range(n_runs))

            _pot(pod, dataset_name, n, method_names, reg_metric_names, results, logger)

    # dump error log
    logger.write_log()

    # mean and std over all regression metrics, runs and datasets
    mean_std_statistics(pod)

    # stability scores
    for metric_name, metric in zip(stab_metric_names, stab_metrics):
        metric(pod)

    return pod
