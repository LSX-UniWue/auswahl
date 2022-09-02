import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .data_handling import DataHandler


def mw_ranking(pod: DataHandler, regression_metric: str, greater_better: bool = False, significance: float = 0.05):
    """Calculates a partial order of methods across all datasets and number of features to be selected for the given
    regression metric. The Mann-Whitney U-test is used to assess statistical significance of differences in regression
    performance.

    Parameters
    ----------
    pod: DataHandler
        BenchmarkPOD object returned by the benchmarking function

    regression_metric: str
        Name of the regression metric w.r.t which the partial order of methods will be calculated

    greater_better: bool, default=False
        Flag, indicating the polarization of the regression metric. Default is the assumption that a smaller value
        in the metric is better

    significance: float, default=0.05
        Significance threshold. A difference in regression performance is considered statistically significant, if
        its p-value is below this threshold

    Returns
    -------
    tuple: (strata, pair_scores)
        strata: partial ordering
        pair_scores: pandas.DataFrame indicating how often a method has been statistically significantly better
                    than other methods, across datasets and number of features to be selected
    """

    data = pod.get_regression_data(reg_metric=regression_metric, item='samples').to_numpy()
    data = np.reshape(data, newshape=(len(pod.methods), pod.n_datasets, len(pod.n_features), pod.n_runs))
    # move methods and samples into the last axes
    data = np.transpose(data, axes=(1, 2, 0, 3))

    # expand axes to enable a broadcasted calculation of the mann-whitney u-statistic
    sample_data = np.expand_dims(data, axis=3)
    testing_data = np.expand_dims(data, axis=2)

    _, p = mannwhitneyu(sample_data, testing_data, alternative=('greater' if greater_better else 'less'), axis=4)

    # reject null hypothesis, accepting alternative (better regression performance)
    p = p < significance

    # yield scores in method-method comparison across all dataset and n_features to be selected
    p = np.sum(p, axis=(0, 1))

    # match each method against each other
    p_t = np.transpose(p, axes=(1, 0))
    pair_comparisons = p > p_t

    # calculate a topological ordering
    strata = []
    method_names = np.array(pod.methods)
    while True:
        # add new stratum
        layer = np.nonzero(np.all(np.logical_not(pair_comparisons), axis=0))[0]
        strata.append(method_names[layer].tolist())

        pair_comparisons = np.delete(pair_comparisons, layer, axis=1)
        pair_comparisons = np.delete(pair_comparisons, layer, axis=0)
        method_names = np.delete(method_names, layer)

        if pair_comparisons.size == 0:
            break

    return strata, pd.DataFrame(p, index=pod.methods, columns=pod.methods)
