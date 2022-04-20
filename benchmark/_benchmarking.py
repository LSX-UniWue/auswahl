import numpy as np
from numpy.random import RandomState
from typing import Union, List, Tuple
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn import clone
from auswahl import PointSelector, IntervalSelector
import timeit as tt

class BenchmarkPOD:

    """
        TODO: extension
        Plain Old Data entity for the exchange of data between benchmarker, user and analytics consumers
    """

    def __init__(self):
        self.data = dict()

    def register(self, method_key: str, **kwargs):
        if method_key not in self.data:
            self.data[method_key] = dict()
        for key in kwargs.keys():
            if key not in self.data[method_key]:
                self.data[method_key][key] = []
            self.data[method_key][key].append(kwargs[key])

    def register_metrics(self, method_key, metrics: List[Tuple[str, float]]):
        if method_key not in self.data:
            self.data[method_key] = dict()
        if 'metrics' not in self.data[method_key]:
            self.data[method_key]['metrics'] = dict()
        for metric_name, value in metrics:
            if metric_name not in self.data[method_key]['metrics']:
                self.data[method_key]['metrics'][metric_name] = []
            self.data[method_key]['metrics'][metric_name].append(value)



def benchmark(data_x: np.array,
              data_y: np.array,
              n_runs: int,
              train_size: Union[int, float],
              test_model, # TODO: type information
              reg_metrics,  # TODO: type information
              methods: List[Union[PointSelector, IntervalSelector]],
              random_state: Union[int, RandomState]):  # TODO: possibly further arguments are to be added

    # TODO: ensure, that not a mixed list of PointSelectors and IntervalSelectors is
    # Different interfaces for both? --> assume for now PointSelectors

    # Extend the methods by their class name (sorting not really necessary), should it be allowed to pass several
    # instances of the same class -> different parameterizations?
    methods = sorted([(type(method).__name__, method) for method in methods], key=lambda tup: tup[0])
    metrics = [(metric.__name__, metric) for metric in (reg_metrics if type(reg_metrics) == list else [reg_metrics])]

    random_state = check_random_state(random_state)

    pod = BenchmarkPOD()
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

            metrics = []
            for metric_name, metric in reg_metrics:
                metrics.append((metric_name, metric(test_y, prediction)))
            pod.register(method, time=exec_time, support=support)
            pod.register_metrics(method, metrics)