import numpy as np
from scipy.stats import mannwhitneyu

from ._data_handling import BenchmarkPOD

def mw_u_test(pod: BenchmarkPOD, metric_name: str, greater_better=False):

    """
        Calculates for the given metric the Mann-Whitney U-test.
        For each method the probability of being better, given the metric, than the other methods.

        Parameters
        ----------
        pod : BenchmarkPOD
            data container produced by benchmarking

        metric_name : str
            identifier of the metric in the BenchmarkPOD object

        greater_better : bool, default=False
            indicator whether a greater value of the metric implies a better performance

    """
    data = []
    methods = pod.get_methods()
    for method in methods:
       data.append(np.array(pod.get_item(method, 'metrics', metric_name, 'samples')))
    data = np.stack(data)

    # mask = np.arange(len(methods))
    for i, method in enumerate(methods):
        # _, p = mannwhitneyu(data[i], np.compress(mask != i, data, axis=0))
        _, p = mannwhitneyu(data[i], data, alternative=('greater' if greater_better else 'smaller'), axis=1)
        pod.register(method, 'metrics', metric_name, mw_u_test=p)


