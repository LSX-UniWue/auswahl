import numpy as np
import matplotlib.patches as mpatches

from typing import List
from matplotlib import pyplot as plt

from ._data_handling import BenchmarkPOD


def plot_score_stability_box(pod: BenchmarkPOD,
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

    for i, method in enumerate(pod.get_methods()):
        plotting_kwargs = dict()
        for entity in entities:
            plotting_kwargs[entity] = dict(color=colors[i])
        score = pod.get_item(method, 'metrics', regression_metric, 'samples')
        stability = pod.get_item(method, stability_metric)
        ax.boxplot(score, positions=stability, whis=(0, 100), manage_ticks=False, **plotting_kwargs)
        legend_handles.append(mpatches.Patch(color=colors[i], label=method))

    ax.set_title("Score-Stability plot")
    ax.set_xlabel("stability score")
    ax.set_ylabel(regression_metric)
    ax.legend(handles=legend_handles)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

