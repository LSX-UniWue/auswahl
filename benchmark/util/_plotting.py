import numpy as np
import matplotlib.patches as mpatches

from typing import List
from matplotlib import pyplot as plt

from ._data_handling import BenchmarkPOD


def plot_score_stability_box(pod: BenchmarkPOD,
                             n_features: int,
                             stability_metric: str,
                             regression_metric: str,
                             save_path: str = None):
    """
        Plotting a boxplot for the benchmarked methods displaying
            * the mean regression score on the y-axis
                * mean regression value
                * (25,75) IQR as box
                * (0, 100) range as whiskers
            * the stability score on the x-axis

        Parameters
        ----------

        pod : BenchmarkPOD
            data container produced by benchmarking

        n_features: int
            number of features, which were to be selected by the algorithms

        stability_metric : str
            identifier of the stability metric to be plotted in the pod

        regression_metric : str
            identifier of the regression metric to be plotted in the pod
    """

    # TODO: color strategies
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    entities = ['boxprops', 'medianprops', 'flierprops', 'capprops', 'whiskerprops']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    legend_handles = []

    for i, method in enumerate(pod.methods):
        plotting_kwargs = dict()
        for entity in entities:
            plotting_kwargs[entity] = dict(color=colors[i])
        score = pod.get_regression_data(method=method, n_features=n_features, reg_metric=regression_metric, item='samples')
        stability = pod.get_stability_data(method=method, n_features=n_features, stab_metric=stability_metric)
        ax.boxplot(score, positions=[stability], whis=(0, 100), widths=0.05, manage_ticks=False, **plotting_kwargs)
        legend_handles.append(mpatches.Patch(color=colors[i], label=method))

    ax.set_title("Score-Stability plot")
    ax.set_xlabel("stability score")
    ax.set_ylabel(regression_metric)
    ax.legend(handles=legend_handles)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_performance_series(pod: BenchmarkPOD,
                            regression_metric: str,
                            save_path: str = None):

    fig, ax = plt.subplots(figsize=(7, 4))
    x_coords = pod.n_features
    markers = [c for c in ".,ov^<>12348sp*hH+xDd|"]
    for i, method in enumerate(pod.methods):

        means = pod.get_regression_data(method=method,
                                        reg_metric=regression_metric,
                                        item='mean')#.groupby(level=0)
        #means = samples.mean().to_numpy()
        #mins = samples.min().to_numpy()
        #max = samples.max().to_numpy()
        #ranges = np.stack([mins, max])

        ax.scatter(x_coords, means, marker=markers[i])

    plt.show()





