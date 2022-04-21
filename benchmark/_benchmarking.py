import numpy as np
import timeit as tt

from numpy.random import RandomState
from typing import Union, List
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn import clone

from auswahl import PointSelector, IntervalSelector
from benchmark.util._data_handling import BenchmarkPOD


def benchmark(data_x: np.array,
              data_y: np.array,
              n_runs: int,
              train_size: Union[int, float],
              test_model, # TODO: type information
              reg_metrics,  # TODO: type information,
              stability_metrics, # TODO: type information
              methods: List[Union[PointSelector, IntervalSelector]],
              random_state: Union[int, RandomState]):  # TODO: possibly further arguments are to be added

    # TODO: ensure, that not a mixed list of PointSelectors and IntervalSelectors is provided
    # Different interfaces for both? --> assume for now PointSelectors

    # Extend the methods by their class name (sorting not really necessary), should it be allowed to pass several
    # instances of the same class -> different parameterizations?
    methods = sorted([(type(method).__name__, method) for method in methods], key=lambda tup: tup[0])
    metrics = [(metric.__name__, metric) for metric in (reg_metrics if type(reg_metrics) == list else [reg_metrics])]

    random_state = check_random_state(random_state)

    pod = BenchmarkPOD()
    pod.register_meta(n_samples=data_x.shape[0], n_wavelengths=data_x.shape[1])

    # coarse and tentative benchmarking draft
    # TODO: parallelization with argument n-jobs
    for r in range(n_runs):
        seed = random_state.randint(0, 1000000)
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=train_size, random_state=seed)
        # TODO: possibly a reset of methods necessary?
        for method in methods:
            # TODO: set the random_state with the seed of the regressor before fitting
            exec_time = tt.timeit(lambda: method.fit(train_x, train_y))
            support = method.get_support()

            test_regressor = clone(test_model)
            test_regressor.fit(train_x[:, support])
            prediction = test_regressor.predict(test_x)

            for metric_name, metric in reg_metrics:
                pod.register(method, 'metrics', metric_name, samples=metric(test_y, prediction))

            pod.register(method, time=exec_time, support=support)

    return pod