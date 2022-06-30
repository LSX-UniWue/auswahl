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

    def log_error(self, severity: str, run_index: int, seed: int, method_name: str, during: str, exception: Exception):
        self.log[f'error {len(self.log) + 1}'] = {'dataset': self.dataset,
                                                  'severity': severity,
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


def _check_positive_integer(x):
    if not isinstance(x, int):
        raise ValueError(f'The specification of features requires integers. Got {type(x)}')
    if x <= 0:
        raise ValueError(f'The specification of features requires positive integers. Got {x}')


def _check_diploid_positive_integer_tuple(x):
    if len(x) != 2:
        raise ValueError("Feature specification with tuples requires a tuple of length 2.")
    v1, v2 = x
    _check_positive_integer(v1)
    _check_positive_integer(v2)
    return x


def _check_feature_consistency(methods, features):

    # make a list
    if not isinstance(features, list):
        features = [features]

    # check presence of an IntervalSelector and non-interval-like feature specification
    contains_interval_selector = np.any([isinstance(method, IntervalSelector) for method in methods])
    contains_non_tuple = np.any([isinstance(item, int) for item in features])

    if contains_interval_selector and contains_non_tuple:
        raise ValueError("An IntervalSelector has been passed to benchmarking. The specification of n_intervals "
                         "and interval_width as features is mandatory")
    feature_keys = []
    if contains_interval_selector:
        # resolve tuples to keys
        features = [_check_diploid_positive_integer_tuple(f) for f in features]
        features = sorted(features, key=lambda x: x[0] * x[1])
        for tup in features:
            feature_keys.append(f'{tup[0]}/{tup[1]}')
    else:
        # resolve possible tuples (PointSelectors only)
        for i, f in enumerate(features):
            if isinstance(f, tuple):
                _check_diploid_positive_integer_tuple(f)
                feature_keys.append(str(f[0] * f[1]))
                features[i] = f[0] * f[1]
            else:
                _check_positive_integer(f)
                feature_keys.append(str(f))
        feature_keys = sorted(feature_keys, key=lambda x: int(x))
        feature = sorted(features)

    # possible resolve duplicates
    features = list(dict.fromkeys(features))
    feature_keys = list(dict.fromkeys(feature_keys))
    return features, feature_keys


def _parameterize(methods, features):
    """
            Reconfigure parameterization of methods.

        Parameters
        ----------
        method: List[Union[PointSelector, IntervalSelector]]
            method to be reparameterized
        features: str
            identifer for the features to be selected in this parameterization

    """
    for method in methods:
        if isinstance(method, PointSelector):
            if isinstance(features, tuple):
                method.set_n_features(features[0] * features[1])
            else:
                method.set_n_features(features)
        else:
            method.set_interval_params(features[0], features[1])


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
        raise ValueError(f'The names in {identifier} need to be unique')


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

    for dataset in data:
        if not isinstance(dataset, (list, tuple)):
            raise ValueError(f'Expected {dataset} to be of type list or tuple')
        if len(dataset) != 4:
            raise ValueError(f'The dataset specification requires three fields: (x, y, name, train_size)')
        if not (isinstance(dataset[0], np.ndarray) and isinstance(dataset[1], np.ndarray)):
            raise ValueError(f'The first two elements in the dataset specification (x, y, name, train_size) need to be of type np.array ')
        if not isinstance(dataset[2], str):
            raise ValueError(f'The name of a dataset is required to be of type str. Got {type(dataset[2])}')
        if not isinstance(dataset[3], float):
            raise ValueError(f'The share of training data is required to be of type float. Got {type(dataset[3])}')

    return zip(*data)


def _check_train_size(train_sizes, data, dataset_names):

    """
        Sanitize training sizes passed to function benchmarking. Check the proper range and scales
        w.r.t. the dataset sizes.

        Parameters
        ----------
        train_sizes: List[float]
            train_size passed to function benchmark
        data: List[np.array]
            datasets passed to function benchmark
    """

    # Check proper range of the sizes
    for size in train_sizes:
        if size <= 0 or size >= 1:
            raise ValueError(f'train_size expected to be in ]0,1[. Got {size}')

    # Check if train_sizes leave a non-empty set of test data for each dataset
    for i in range(len(train_sizes)):
        x = data[i]
        if int(x.shape[0] * (1 - train_sizes[i])) == 0:
            raise ValueError(f'Given the size of the dataset {dataset_names[i]}, the specified train_size {train_sizes[i]} leaves '
                             f'an empty test set.')


def _check_n_runs(n_runs):

    if not isinstance(n_runs, int):
        raise ValueError(f'n_runs is required to be an integer')
    if n_runs <= 0:
        raise ValueError(f'n_runs is required to be positive')


def _copy_methods(methods, joblib_mem_segregation=True):
    if joblib_mem_segregation:
        return methods
    else:
        return [copy.deepcopy(method) for method in methods]


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


def _benchmark_parallel(x: np.array, y: np.array, train_size: float,
                        methods, method_names, reg_metrics, seed: int, run_index: int):

    methods = _copy_methods(methods)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=train_size, random_state=seed)

    results = dict()
    for method_name, method in zip(method_names, methods):
        results[method_name] = dict()
        _reseed(method, seed)
        # fit feature selector
        try:
            start = time.process_time()
            method.fit(train_x, train_y)
            end = time.process_time()
        except Exception as e:
            results[method_name]['exception'] = (e, run_index, seed, "Fitting of Selector")
            continue
        # fit test model
        support = method.get_support(indices=True)
        model = method.best_model_
        try:
            model.fit(train_x[:, support], train_y)
        except Exception as e:
            results[method_name]['exception'] = (e, run_index, seed, 'Fitting of test model')
            continue
        # make test model prediction
        try:
            prediction = model.predict(test_x[:, support])
        except Exception as e:
            results[method_name]['exception'] = (e, run_index, seed, 'Prediction of test model')
            continue
        # execute stability metrics
        metrics = []
        for metric in reg_metrics:
            try:
                metrics.append(metric(test_y, prediction))
            except Exception as e:
                metrics.append(np.nan)
                metric_errors = results[method_name].setdefault('metric_exception', [])
                metric_errors.append((e, run_index, seed, f'Evaluation of: {metric.__name__}'))

        results[method_name]['metrics'] = metrics
        results[method_name]['exec'] = end-start
        results[method_name]['selection'] = support
        results[method_name]['run_index'] = run_index  # retain association of threads to runs

    return results


def _pot(pod, dataset_name, feature, methods_names, reg_metrics_names, results, logger: ErrorLogger):
    for result in results:
        for method in methods_names:
            if 'exception' not in result[method].keys():
                run_index = result[method]['run_index']
                pod.register_measurement(result[method]['exec'], dataset_name, method,
                                         feature, 'samples', run_index)
                pod.register_selection(dataset_name, method,
                                       feature, run_index, result[method]['selection'])
                for j, metric in enumerate(reg_metrics_names):
                    pod.register_regression(result[method]['metrics'][j], dataset_name, method,
                                            feature, metric, 'samples', run_index)
            else:
                exception, run_index, seed, during = result[method]['exception']
                logger.log_error(severity='fatal', run_index=run_index, seed=seed,
                                 method_name=method, during=during, exception=exception)

            if 'metric_exception' in result[method].keys():
                for ex in result[method]['metric_exception']:
                    exception, run_index, seed, during = ex
                    logger.log_error(severity='metric error', run_index=run_index, seed=seed,
                                     method_name=method, during=during, exception=exception)


def benchmark(data: List[Tuple[np.array, np.array, str, float, BaseEstimator]],
              features: List[Union[int, Tuple[int, int]]],
              n_runs: int,
              methods: List[Union[PointSelector, IntervalSelector, Tuple[Union[PointSelector, IntervalSelector], str]]],
              reg_metrics: List[Callable[[np.ndarray, np.ndarray], float]],
              random_state: Union[int, RandomState],
              stab_metrics: List[Callable[[np.ndarray, np.ndarray], float]] = [],
              n_jobs: int = 1,
              error_log_file: str = "./error_log.txt",
              verbose: bool = True):

    """
        Function performing benchmarking of Interval- and PointSelector feature selectors across
        different datasets and different parameterizations of the selectors.

        Parameters
        ----------
        data
        features
        n_runs
        reg_metrics
        stab_metrics
        methods
        random_state
        n_jobs
        error_log_file
        verbose

        Returns
        -------
        pod: BenchmarkPOD
            TODO
    """
    speaker = Speaker(verbose)
    logger = ErrorLogger(log_file=error_log_file)

    features, feature_keys = _check_feature_consistency(methods, features)

    xs, ys, dataset_names, train_sizes = _check_datasets(data)
    reg_metrics, reg_metric_names = _unpack_metrics(reg_metrics, compulsory=True)
    stab_metric, stab_metric_names = _unpack_metrics(stab_metrics)
    methods, method_names = _unpack_methods(methods)

    _check_name_uniqueness(dataset_names, "datasets")
    _check_name_uniqueness(method_names,  "methods")
    _check_name_uniqueness(reg_metric_names, "reg_metrics")
    _check_name_uniqueness(stab_metric_names, "stab_metrics")

    _check_train_size(train_sizes, xs, dataset_names)
    _check_n_runs(n_runs)

    # configure the methods to have a single-thread operation (parallelization is exploited at the benchmarking level)
    _drain_threads(methods)

    random_state = check_random_state(random_state)

    pod = BenchmarkPOD(dataset_names,
                       method_names,
                       features,
                       feature_keys,
                       reg_metric_names,
                       stab_metric_names,
                       n_runs
                       )

    pod.register_meta(data)

    # pregenerate seeds, such that the seeding of data splits does not depend on the arguments except random_state
    run_seeds = random_state.randint(0, 1000000, size=n_runs)

    with Parallel(n_jobs=n_jobs) as parallel:
        for d in range(len(dataset_names)):
            speaker.announce(level=0, message=f'Started benchmark for dataset {dataset_names[d]}')
            for i, n in enumerate(features):
                speaker.announce(level=1, message=f'Started cycle with {n} features to select')
                logger.set_meta(dataset=dataset_names[d], features=n)

                _parameterize(methods, n)
                results = parallel(delayed(_benchmark_parallel)(xs[d], ys[d], train_sizes[d],
                                                                methods, method_names, reg_metrics,
                                                                run_seeds[r], r)
                                   for r in range(n_runs))

                # insert the results of the processes into the BenchmarkPOD object or the error log
                _pot(pod, dataset_names[d], features[i], method_names, reg_metric_names, results, logger)

    # dump error log
    logger.write_log()

    # mean and std over all regression metrics, runs and datasets
    mean_std_statistics(pod)

    # stability scores
    for metric_name, metric in zip(stab_metric_names, stab_metrics):
        metric(pod)

    return pod
