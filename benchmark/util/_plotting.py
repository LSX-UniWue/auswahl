
import numpy as np
import matplotlib.patches as mpatches

from typing import List, Union, Literal
from matplotlib import pyplot as plt

from ._data_handling import BenchmarkPOD

def _box_plot(title: str,
              x_label: str,
              y_label: str,
              y_data: List[List[float]],
              x_data: List[float],
              legend: List[str],
              save_path: str = None):

    # TODO: color strategies
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    entities = ['boxprops', 'medianprops', 'flierprops', 'capprops', 'whiskerprops']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    legend_handles = []

    if legend is None:
        plotting_kwargs = dict()
        for entity in entities:
            plotting_kwargs[entity] = dict(color='k')

    for i, data in enumerate(y_data):
        if legend is not None:
            plotting_kwargs = dict()
            for entity in entities:
                plotting_kwargs[entity] = dict(color=colors[i])

        ax.boxplot(data, positions=[x_data[i]], whis=(0, 100), widths=0.05, manage_ticks=False, **plotting_kwargs)
        if legend is not None:
            legend_handles.append(mpatches.Patch(color=colors[i], label=legend[i]))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if legend is not None:
        ax.legend(handles=legend_handles)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def _errorbar_plot(title: str,
                   x_label: str,
                   y_label: str,
                   y_data: List[List[float]],
                   y_max: List[List[float]],
                   y_min: List[List[float]],
                   x_data: List[List[Union[float, int]]],
                   legend: List[str],
                   plot_lines: bool = True,
                   save_path: str = None):

    # TODO: color strategies
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    markers = [c for c in ".ov^<>12348sp*hH+xDd|"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    legend_handles = []

    for i, y_data in enumerate(y_data):
        ax.errorbar(x_data[i] if len(x_data) > 1 else x_data[0],
                    y_data,
                    yerr=[y_min[i], y_max[i]],
                    color=colors[i],
                    marker=markers[i],
                    linestyle='dotted' if plot_lines else 'none')
        legend_handles.append(mpatches.Patch(color=colors[i], label=legend[i]))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(handles=legend_handles)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_score_stability_box(pod: BenchmarkPOD,
                             dataset: str,
                             n_features: int,
                             stability_metric: str,
                             regression_metric: str,
                             methods: Union[str, List[str]] = None,
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

        dataset: str
            dataset for which the data is to be plotted

        n_features: int
            number of features, which were to be selected by the algorithms

        stability_metric : str
            identifier of the stability metric to be plotted in the pod

        regression_metric : str
            identifier of the regression metric to be plotted in the pod

        methods : str or list of str, default=None
            identifers of methods for which the data is to be plotted. If None, all available methods are plotted

        save_path: str, default=None
            path at which the plot is stored. If None, the plot is just displayed
    """

    regression_scores = []
    stability_scores = []

    if methods is None:
        methods = pod.methods
    elif type(methods) == str:
        methods = [methods]

    for i, method in enumerate(methods):
        regression_scores.append(pod.get_regression_data(method=method,
                                                         n_features=n_features,
                                                         reg_metric=regression_metric,
                                                         item='samples'))
        stability_scores.append(pod.get_stability_data(dataset=dataset,
                                                       method=method,
                                                       n_features=n_features,
                                                       stab_metric=stability_metric))
    _box_plot("Regression-Stability-Plot",
              stability_metric,
              regression_metric,
              regression_scores,
              stability_scores,
              pod.methods,
              save_path)



def plot_exec_time(pod: BenchmarkPOD,
                   dataset: str,
                   methods: Union[str, List[str]] = None,
                   n_features: Union[int, List[int]] = None,
                   item: Literal['mean', 'median'] = 'mean',
                   save_path: str = None):

    exec_times = pod.get_measurement_data(dataset=dataset,
                                          method=methods,
                                          n_features=n_features,
                                          item=item).to_numpy().tolist()
    exec_mins = pod.get_measurement_data(dataset=dataset,
                                         method=methods,
                                         n_features=n_features,
                                         item='min').to_numpy().tolist()
    exec_max = pod.get_measurement_data(dataset=dataset,
                                        method=methods,
                                        n_features=n_features,
                                        item='max').to_numpy().tolist()

    _errorbar_plot(f'Execution time: {item} and ranges',
                   "n_features",
                   "Execution time [s]",
                   exec_times,
                   exec_max,
                   exec_mins,
                   [n_features if n_features is not None else pod.n_features],
                   methods if methods is not None else pod.methods,
                   save_path)


def plot_performance_series(pod: BenchmarkPOD,
                            dataset: str,
                            regression_metric: str,
                            methods: Union[str, List[str]] = None,
                            n_features: Union[int, List[int]] = None,
                            item: Literal['mean', 'median'] = 'mean',
                            save_path: str = None):

    reg_scores = pod.get_regression_data(method=methods,
                                         n_features=n_features,
                                         dataset=dataset,
                                         reg_metric=regression_metric,
                                         item=item).to_numpy().tolist()
    reg_mins = pod.get_regression_data(method=methods,
                                       dataset=dataset,
                                       n_features=n_features,
                                       reg_metric=regression_metric,
                                       item='min').to_numpy().tolist()
    reg_max = pod.get_regression_data(method=methods,
                                      dataset=dataset,
                                      n_features=n_features,
                                      reg_metric=regression_metric,
                                      item='max').to_numpy().tolist()

    # calculate offset x coordinates
    x_coords = []
    ticks = np.array(n_features if n_features is not None else pod.n_features)
    n_methods = len(methods if methods is not None else pod.methods)
    for i in range(n_methods):
        x_coords.append(-0.1 + ticks + (0.2/(n_methods - 1)) * i)


    _errorbar_plot(f'Regression performance: {item} and range on dataset {dataset}',
                   "n_features",
                   regression_metric,
                   reg_scores,
                   reg_max,
                   reg_mins,
                   x_coords,
                   methods if methods is not None else pod.methods,
                   plot_lines=False,
                   save_path=save_path
                   )

